import logging
import os
import re
import time
from collections import defaultdict
from contextlib import contextmanager
from functools import reduce
from pathlib import Path
from textwrap import dedent
from typing import Union, List, Tuple, Type, Dict, Set, Callable, Optional
from uuid import uuid4

import graphviz
import networkx as nx
import pandas as pd
import py2neo
import textdistance
from networkx import NetworkXNoPath, NodeNotFound
from networkx.drawing.nx_pydot import to_pydot
from operator import mul
from py2neo import ClientError, DatabaseError
from tqdm import tqdm

from .file import File, HDU
from .graph import Graph, _convert_datatypes
from .hierarchy import Multiple, Hierarchy, Graphable, OneOf
from .path_finding import HierarchyGraph, get_all_class_bases
from .readquery import Query
from .readquery.exceptions import UserError, CardinalityError
from .readquery.results import RowParser
from .utilities import make_plural, make_singular
from .writequery import Unwind

CONSTRAINT_FAILURE = re.compile(r"already exists with label `(?P<label>[^`]+)` and property "
                                r"`(?P<idname>[^`]+)` = (?P<idvalue>[^`]+)$", flags=re.IGNORECASE)


class ConnectionTimeOutError(ConnectionError):
    pass


def process_neo4j_error(data: 'Data', file: File, msg):
    matches = CONSTRAINT_FAILURE.findall(msg)
    if not len(matches):
        return  # cannot help
    label, idname, idvalue = matches[0]
    # get the node properties that already exist
    extant = data.graph.neograph.evaluate(f'MATCH (n:{label} {{{idname}: {idvalue}}}) RETURN properties(n)')
    fname = data.graph.neograph.evaluate(f'MATCH (n:{label} {{{idname}: {idvalue}}})-[*]->(f:File) return f.fname limit 1')
    idvalue = idvalue.strip("'").strip('"')
    file.data = data
    obj = [i for i in data.hierarchies if i.__name__ == label][0]
    instance_list = getattr(file, obj.plural_name)
    new = {}
    if not isinstance(instance_list, (list, tuple)):  # has an unwind table object
        new_idvalue = instance_list.identifier
        if isinstance(new_idvalue, Unwind):
            # find the index in the table and get the properties
            filt = (new_idvalue.data == idvalue).iloc[:, 0]
            for k in extant.keys():
                if k == 'id':
                    k = idname
                value = getattr(instance_list, k, None)
                if isinstance(value, Unwind):
                    table = value.data.where(pd.notnull(value.data), 'NaN')
                    new[k] = str(table[k][filt].values[0])
                else:
                    new[k] = str(value)
        else:
            # if the identifier of this object is not looping through a table, we cant proceed
            return
    else:  # is a list of non-table things
        found = [i for i in instance_list if i.identifier == idvalue][0]
        for k in extant.keys():
            value = getattr(found, k, None)
            new[k] = value
    comparison = pd.concat([pd.Series(extant, name='extant'), pd.Series(new, name='to_add')], axis=1)
    filt = comparison.extant != comparison.to_add
    filt &= ~comparison.isnull().all(axis=1)
    where_different = comparison[filt]
    logging.exception(f"The node (:{label} {{{idname}: {idvalue}}}) tried to be created twice with different properties.")
    logging.exception(f"{where_different}")
    logging.exception(f"filenames: {fname}, {file.fname}")


def get_all_subclasses(cls: Type[Graphable]) -> List[Type[Graphable]]:
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses


def find_children_of(parent):
    hierarchies = get_all_subclasses(Hierarchy)
    children = set()
    for h in hierarchies:
        if len(h.parents):
            if any(p is parent if isinstance(p, type) else p.node is parent for p in h.parents):
                children.add(h)
    return children


class IndirectAccessError(Exception):
    pass


class MultiplicityError(Exception):
    pass


def is_multiple_edge(graph, x, y):
    return not graph.edges[(x, y)]['multiplicity']

def expand_template_relation(relation, allowed_hiers: Set[Type[Hierarchy]]):
    """
    Returns a list of relations that relate to each non-template class
    e.g.
    >>> expand_template_relation(Multiple(L1StackedSpectrum))
    [Multiple(L1SingleSpectrum), Multiple(L1StackSpectrum), Multiple(L1SuperstackSpectrum)]
    """
    if not relation.node.is_template:
        return [relation]
    subclasses = [cls for cls in get_all_subclasses(relation.node) if not cls.is_template and cls in allowed_hiers]
    return [Multiple(subclass, 0, relation.maxnumber, relation.constrain, relation.relation_idname, relation.one2one) for subclass in subclasses]


def add_relation_graph_edge(graph, parent, child, relation: Multiple, allowed_hiers: Set[Type[Hierarchy]]):
    """
    if an object of type O requires n parents of type P then this is equivalent to defining that instances of those behave as:
        P-(n)->O (1 object of type O has n parents of type P)
    it implicitly follows that:
        O--(m)--P (each of object's parents of type P can be used by an unknown number `m` of objects of type O = many to one)
    if an object of type O requires n children of type C then this is equivalent to defining that instances of those behave as:
        O-(n)->C (1 object has n children of type C)
        it implicitly follows that:
            child-[m]->Object (each child has m parents of type O)
    """
    relation.instantate_node()
    child_defines_parents = relation.node is parent
    for relation in expand_template_relation(relation, allowed_hiers):
        relation.instantate_node()
        # only parent-->child is in the database
        relstyle = 'solid' if relation.maxnumber == 1 else 'dashed'
        if parent is child:
            for a, b in [(parent, child), (child, parent)]:
                graph.add_edge(a, b, singular=relation.maxnumber == 1,
                               optional=relation.minnumber == 0, style=relstyle, actual_number=1)
        elif child_defines_parents:  # i.e. parents = [...] is set in the class for this object
            # child instance has n of type Parent, parent instance has unknown number of type Child
            parent = relation.node  # reset from new relations
            graph.add_edge(child, parent, singular=relation.maxnumber == 1,
                           optional=relation.minnumber == 0, style=relstyle, actual_number=1)
            if relation.one2one:
                graph.add_edge(parent, child, singular=True, optional=True, style='solid',
                               relation=relation, actual_number=1)
            else:
                graph.add_edge(parent, child, singular=False, optional=True, style='dotted',
                               relation=relation, actual_number=1)
        else:  # i.e. children = [...] is set in the class for this object
            # parent instance has n of type Child, each child instance has one of type Parent
            child = relation.node  # reset from new relations
            graph.add_edge(parent, child, singular=relation.maxnumber == 1,
                           optional=relation.minnumber == 0,
                           relation=relation, style=relstyle, actual_number=1)
            graph.add_edge(child, parent, singular=True, optional=True, style='solid', actual_number=1)


def make_relation_graph(hierarchies: Set[Type[Hierarchy]]):
    graph = nx.DiGraph()
    for h in hierarchies:
        if h not in graph.nodes:
            graph.add_node(h)
        for child in h.children:
            rel = child if isinstance(child, Multiple) else OneOf(child)
            child = child.node if isinstance(child, Multiple) else child
            add_relation_graph_edge(graph, h, child, rel, hierarchies)
        for parent in h.parents:
            rel = parent if isinstance(parent, Multiple) else OneOf(parent)
            parent = parent.node if isinstance(parent, Multiple) else parent
            add_relation_graph_edge(graph, parent, h, rel, hierarchies)
    return graph

def hierarchies_from_hierarchy(hier: Type[Hierarchy], done=None, templates=False) -> Set[Type[Hierarchy]]:
    if done is None:
        done = []
    hierarchies = set()
    todo = {h.node if isinstance(h, Multiple) else h for h in hier.parents + hier.children + hier.produces}
    if not templates:
        todo = {h for h in todo if not h.is_template}
    else:
        todo.update({h for h in todo for hh in get_all_class_bases(h) if issubclass(hh, Hierarchy)})
    for new in todo:
        if isinstance(new, Multiple):
            new.instantate_node()
            h = new.node
        else:
            h = new
        if h not in done and h is not hier:
            hierarchies.update(hierarchies_from_hierarchy(h, done, templates))
            done.append(h)
    hierarchies.add(hier)
    return hierarchies

def hierarchies_from_files(*files: Type[File], templates=False) -> Set[Type[Hierarchy]]:
    hiers = {h.node if isinstance(h, Multiple) else h for file in files for h in file.children + file.produces}
    if not templates:
        hiers = {h for h in hiers if not h.is_template}
    else:
        hiers.update({h for h in hiers for hh in get_all_class_bases(h) if issubclass(hh, Hierarchy)})
    hiers.update(set(files))
    return reduce(set.union, map(hierarchies_from_hierarchy, hiers))

def make_arrows(path, forwards: List[bool], descriptors=None):
    descriptors = [descriptors]*len(forwards) if not isinstance(descriptors, (list, tuple)) else descriptors
    descriptors = [f":{descriptor}" if descriptor is not None else "" for descriptor in descriptors]
    assert len(forwards) == len(path) - 1
    forward_arrow = '-[{name}{descriptor}]->'
    backward_arrow = '<-[{name}{descriptor}]-'
    nodes = list(map('(:{})'.format, [p.__name__ for p in path[1:]]))
    path_list = ['({in_var})']
    for i, (node, forward) in enumerate(zip(nodes, forwards)):
        arrow = forward_arrow if forward else backward_arrow
        if i == len(forwards) - 1:
            arrow = arrow.format(name='{name}', descriptor=descriptors[i])
        else:
            arrow = arrow.format(name="", descriptor=descriptors[i])
        path_list.append(arrow)
        path_list.append(node)
    path_list[-1] = path_list[-1].replace('(:', '({out_var}:')
    return ''.join(path_list)


def plot_graph(G, fname, format):
    return graphviz.Source(to_pydot(G).to_string()).render(fname, format=format)


class Data:
    filetypes = []
    extra_hierarchies = []

    def __new__(cls, *args, **kwargs):
        from fastapi.encoders import ENCODERS_BY_TYPE
        ENCODERS_BY_TYPE[cls] = lambda x: x.as_dict()
        return super().__new__(cls)

    def __init__(self, rootdir: Union[Path, str] = None,
                 host: str = None, port=None, dbname=None,
                 password=None, user=None, verbose=False):
        self.verbose = verbose
        self.dbname = dbname or os.getenv('WEAVEIO_DB', 'production')
        self.host = host or os.getenv('WEAVEIO_HOST', '127.0.0.1')
        self.port = port or os.getenv('WEAVEIO_PORT', 7687)
        self.password = password or os.getenv('WEAVEIO_PASSWORD', 'weavepassword')
        self.user = user or os.getenv('WEAVEIO_USER', 'weaveuser')
        rootdir = rootdir or os.getenv('WEAVEIO_ROOTDIR')
        if rootdir is None:
            raise ValueError(f"You must specify WEAVEIO_ROOTDIR as an environment variable or as an argument to Data (rootdir=...)")
        self.rootdir = Path(rootdir)
        self.write_allowed = False
        self.query = Query(self)
        self.rowparser = RowParser(self.rootdir)
        self.filelists = {}
        self.hierarchy_graph = HierarchyGraph()
        self.hierarchy_graph.initialise()
        if self.filetypes:
            self.hierarchies = hierarchies_from_files(*self.filetypes, templates=True)
        else:
            self.hierarchies = set()
        self.hierarchies.update(set(self.extra_hierarchies))
        self.hierarchies.update({hh for h in self.hierarchies for hh in get_all_class_bases(h)})
        self.class_hierarchies = {h.__name__: h for h in self.hierarchies}
        self.singular_hierarchies = {h.singular_name: h for h in self.hierarchies}  # type: Dict[str, Type[Hierarchy]]
        self.plural_hierarchies = {h.plural_name: h for h in self.hierarchies if h.plural_name != 'graphables'}
        self.factor_hierarchies = defaultdict(set)
        for h in self.hierarchies:
            for f in getattr(h, 'products_and_factors', []):
                self.factor_hierarchies[f.lower()].add(h)
            if h.idname is not None:
                self.factor_hierarchies[h.idname].add(h)
        self.factor_hierarchies = dict(self.factor_hierarchies)  # make sure we always get keyerrors when necessary!
        self.factors = set(self.factor_hierarchies.keys())
        self.plural_factors =  {make_plural(f.lower()): f.lower() for f in self.factors}
        self.singular_factors = {f.lower() : f.lower() for f in self.factors}
        self.singular_idnames = {h.idname: h for h in self.hierarchies if h.idname is not None}
        self.plural_idnames = {make_plural(k): v for k,v in self.singular_idnames.items()}
        self.relative_names = defaultdict(dict)
        for h in self.hierarchies:
            for name, relation in h.relative_names.items():
                self.relative_names[name][h.__name__] = relation
        self.relative_names = dict(self.relative_names)
        self.plural_relative_names = {make_plural(name): name for name in self.relative_names}

    def __repr__(self):
        return f'<Data({self.user}@{self.host}[{self.dbname}]:{self.port}:{self.rootdir})>'

    def as_dict(self):
        return {'user': self.user, 'host': self.host, 'dbname': self.dbname, 'port': self.port, 'rootdir': self.rootdir}

    def __dir__(self) -> List[str]:
        neighbors = [i.plural_name for i in self.hierarchies]
        neighbors += [i.lower() for h in self.hierarchies for i in h.products_and_factors]
        neighbors = [i if '[' not in i else f'"{i}"' for i in neighbors]
        return neighbors

    def _ipython_key_completions_(self):
        return [i.strip('"') for i in self.__dir__()]

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value
        if value:
            logging.basicConfig(level=logging.INFO)

    # noinspection PyTypeHints
    def expand_template_object(self, obj: str) -> Set[str]:
        obj = self.singular_hierarchies[self.singular_name(obj)]
        return {h.__name__ for h in self.hierarchies if issubclass(h, obj) and not h.is_template}

    def expand_dependent_object(self, obj: str, *dependencies: str) -> Set[str]:
        obj = self.singular_hierarchies[self.singular_name(obj)]
        dependencies = {self.singular_hierarchies[self.singular_name(o)] for o in dependencies}
        parents = self.parents_of_defined_child(obj)
        expanded_dependencies = set()
        for o in dependencies:
            if o.is_template:
                expanded_dependencies |= set(get_all_subclasses(o))
            else:
                expanded_dependencies.add(o)
        return {o.__name__ for o in parents if o in expanded_dependencies}

    def _path_to_hierarchy(self, from_obj: Type[Hierarchy], to_obj:  Type[Hierarchy], singular: bool):
        """
        When searching for a path, the target is either above or below the source in one direction only
        If the target is defined as a child by another, then the search is redefined as for the predecessors of that child
            i.e. ob.nosses => [ob.l1single_spectra.noss, ob.l1stack_spectra.noss, ob.l1superstack_spectra.noss, ...]
                 this is then an error since it is an ambiguous path
        If the source is defined as a child by another, then additionally, the other node is required to define a path
            i.e. ...noss.obs requires that ... is known.
                 this is therefore an error since it is an ambiguous path
        If force_single, use only single edges
        for {singles, multiples}:
            Find the shortest path in one direction, but search both directions, both are equally valid
            If more than one path is returned, throw an ambiguous path exception

        """
        paths = list(self.hierarchy_graph.find_paths(from_obj, to_obj, singular))
        if not paths:
            if singular:
                raise NetworkXNoPath(f"No singular path found between `{from_obj}` and `{to_obj}`")
            raise NetworkXNoPath(f"No path found between `{from_obj}` and `{to_obj}`")
        paths, edges, reversed = zip(*[(path[::-1], [self.hierarchy_graph.short_edge(e[1], e[0]) for e in es[::-1]], True) if path[0] is to_obj else (path, es, False) for path, es in paths])
        singulars = [self.hierarchy_graph.edge_path_is_singular(es) for es in edges]
        return paths, singulars, reversed


    def parents_of_defined_child(self, potential_child: Type[Hierarchy]) -> Set[Type[Hierarchy]]:
        parents = {h for h in self.hierarchies if potential_child in [c.node if isinstance(c, Multiple) else c for c in h.children]}
        return {p for p in parents if not issubclass(p, File) and p is not potential_child}

    def is_generic_object(self, obj: str) -> bool:
        try:
            h = self.singular_hierarchies[self.singular_name(obj)]
            return h.is_template or self.parents_of_defined_child(h)
        except KeyError:
            return False

    def paths_to_hierarchy(self, from_obj: str, to_obj: str, singular: bool, descriptor=None, return_objs=False):
        a, b = map(self.singular_name, [from_obj, to_obj])
        from_obj, to_obj = self.singular_hierarchies[a], self.singular_hierarchies[b]
        try:
            paths, singulars, reversed = self._path_to_hierarchy(from_obj, to_obj, singular)
            arrows = [make_arrows(path, [not r]*(len(path)-1), descriptor) for path, r in zip(paths, reversed)]
            if return_objs:
                return arrows, singulars, paths
            return arrows, singulars
        except nx.NetworkXNoPath:
            if not singular:
                to = f"multiple `{self.plural_name(b)}`"
            else:
                to = f"only one `{self.singular_name(b)}`"
            from_ = self.singular_name(a.lower())
            raise NetworkXNoPath(f"Can't find a link between `{from_}` and {to}. "
                                f"This may be because it doesn't make sense for `{from_}` to have {to}. "
                                f"Try checking the cardinality of your query.")

    def all_links_to_hierarchy(self, hierarchy: Type[Hierarchy], edge_constraint: Callable[[nx.DiGraph, Tuple], bool]) -> Set[Type[Hierarchy]]:
        hierarchy = self.class_hierarchies[self.class_name(hierarchy)]
        g = self.relation_graphs[-1]
        singles = nx.subgraph_view(g, filter_edge=lambda a, b: edge_constraint(g, (a, b)))
        hiers = set()
        for node in singles.nodes:
            if nx.has_path(singles, hierarchy, node):
                hiers.add(node)
        return hiers

    def all_single_links_to_hierarchy(self, hierarchy: Type[Hierarchy]) -> Set[Type[Hierarchy]]:
        return self.all_links_to_hierarchy(hierarchy, lambda g, e: g.edges[e]['singular'])

    def all_multiple_links_to_hierarchy(self, hierarchy: Type[Hierarchy]) -> Set[Type[Hierarchy]]:
        return self.all_links_to_hierarchy(hierarchy, lambda g, e: not g.edges[e]['singular'])

    def write_cypher(self, collision_manager='track&flag'):
        if self.write_allowed:
            return self.graph.write(collision_manager)
        raise IOError(f"You have not allowed write operations in this instance of data (write=False)")

    @property
    @contextmanager
    def write(self):
        self.write_allowed = True
        yield self.graph
        self.write_allowed = False

    def is_unique_factor(self, name):
        return len(self.factor_hierarchies[name]) == 1

    @property
    def graph(self):
        d = {}
        if self.password is not None:
            d['password'] = self.password
        if self.user is not None:
            d['user'] = self.user
        return Graph(host=self.host, port=self.port, name=self.dbname, write=self.write_allowed, **d)

    def make_constraints_cypher(self) -> Dict[str, List[str]]:
        d = {hierarchy: hierarchy.make_schema() for hierarchy in self.hierarchies}
        d['temporary'] = ['CREATE CONSTRAINT ON (t:TemporaryMerge) ASSERT t.id IS UNIQUE']
        d['dbcreated'] = ['CREATE INDEX n_dbcreated FOR (n:Hierarchy) ON n.`_dbcreated`']
        d['dbupdated'] = ['CREATE INDEX n_dbupdated FOR (n:Hierarchy) ON n.`_dbupdated`']
        return d

    def apply_constraints(self, silent=False):
        if not self.write_allowed:
            raise IOError(f"Writing is not allowed")
        templates = set()
        equivalencies = set()
        for hier, qs in tqdm(self.make_constraints_cypher().items(), desc='applying constraints', disable=silent):
            if not qs:
                templates.add(hier)
            for q in qs:
                try:
                    self.graph.neograph.run(q)
                except py2neo.ClientError as e:
                    if '[Schema.EquivalentSchemaRuleAlreadyExists]' in str(e):
                        equivalencies.add(hier)
                        templates.add(hier)
        total = len(templates)
        templates -= equivalencies
        if len(equivalencies):
            n = f'({len(equivalencies)}/{total})'
            equivalencies = ', '.join(sorted([getattr(t, '__name__', str(t)) for t in equivalencies]))
            logging.info(f'EquivalentSchemaRuleAlreadyExists for {n}: {equivalencies}')
        if len(templates):
            n = f'({len(templates)}/{total})'
            templates = ', '.join(sorted([getattr(t, '__name__', str(t)) for t in templates]))
            logging.info(f'No index/constraint exists for {n}: {templates}')
        self.await_indexes()

    def await_indexes(self, timeout=300):
        logging.info("Waiting for indexes to come online...")
        self.graph.neograph.run(f'call db.awaitIndexes({timeout})')
        logging.info("Indexes are online")

    def drop_all_constraints(self, indexes=True):
        if not self.write_allowed:
            raise IOError(f"Writing is not allowed")
        constraints = self.graph.neograph.run('CALL db.constraints() YIELD name return "DROP CONSTRAINT " + name + ";"')
        for constraint in tqdm(constraints, desc='dropping constraints'):
            self.graph.neograph.run(str(constraint)[1:-1])
        if indexes:
            self.drop_all_indexes()

    def drop_all_indexes(self):
        if not self.write_allowed:
            raise IOError(f"Writing is not allowed")
        indexes = self.graph.neograph.run('CALL db.indexes() yield name, labelsOrTypes where size(labelsOrTypes) > 0 return "DROP INDEX " + name + ";"')
        for index in tqdm(indexes, desc='dropping indexes'):
            self.graph.neograph.run(str(index)[1:-1])

    def get_complete_files(self, *filetypes, return_only_fname=True):
        filetypes = [self.singular_hierarchies[self.singular_name(f)].__name__ for f in filetypes]
        if not filetypes:
            filetypes = ['File']
        r = 'fname' if return_only_fname else 'path'
        q = f"MATCH (f:File) where any(l in labels(f) where l in $filetypes) and f._dbcompleted_complete RETURN DISTINCT f.{r}"
        result = self.graph.execute(q, filetypes=filetypes).to_series(0).values.tolist()
        if return_only_fname:
            return result
        return list(map(lambda x: self.rootdir / x, result))

    def raise_collisions(self):
        """
        returns the properties that would have been overwritten in nodes and relationships.
        """
        node_collisions = self.graph.execute("MATCH (c: _Collision) return c { .*}").to_data_frame()
        rel_collisions = self.graph.execute("MATCH ()-[c: _Collision]-() return c { .*}").to_data_frame()
        return node_collisions, rel_collisions

    def get_state_of_file(self, fname: str) -> int:
        """Returns the state (timestamp) of the creation of the last data product belong to a given file"""
        c = self.data.execute("match (f: File {fname: $fname}) optional match (f)<-[:is_required_by]-(h:Hierarchy) "
                          "where not h:HDU with  h.`_dbcreated` as d order by d desc return d limit 1", fname=fname)
        return c.evaluate()

    def restore_state(self, state: int):
        """
        Restores database to the given state (timestamp). Danger zone here.
        """
        if not self.write_allowed:
            raise IOError(f"Writing is not allowed")
        return self.graph.execute('match (n) where n.`_dbcreated` > $timestamp detach delete n', timestamp=state)

    def mark_batch_complete_query(self, fname, slice, part, total_length, all_parts):
        """
        Mark fname[slice][part] as complete in the db and if all parts are then complete, mark that too
        file._dbcompleted_{part} is a list of rows that have been marked as complete for that part
        file._dbcompleted_{part}_complete = True if that part is completely written
        file._dbcompleted_complete = True all parts are completely written
        part is "None" for files which have no parts
        """
        def get_intervals(v):
            return f"[c in apoc.coll.pairs([0] + [x in range(0, size({v})-2) where {v}[x] <> {v}[x+1] -1 | x+1]) | CASE WHEN c[1] is null THEN [{v}[c[0]],{v}[-1]] ELSE [{v}[c[0]], {v}[c[1]-1]] END]"
        def get_list(v):
            return f"apoc.coll.flatten([i in {v} | range(i[0],i[1])])"
        q = dedent(rf"""
        MATCH (f: File {{fname: $_check_fname}})
        WITH *, apoc.coll.zip(f.`_dbcompleted_{part}_start`, f.`_dbcompleted_{part}_end`) as already_intervals
        WITH *, {get_list('already_intervals')} as already_done
        WITH *, apoc.coll.sort(apoc.coll.toSet(coalesce(already_done, []) + $_check_completed))+[-1] as done
        WITH *, {get_intervals('done')} as done_intervals
        WITH *, CASE WHEN size(done_intervals) = 0 THEN [[0, $_check_total_length-1]] ELSE done_intervals[..-1] END as done_intervals
        SET f.`_dbcompleted_{part}_start` = [i in done_intervals | i[0]]
        SET f.`_dbcompleted_{part}_end` = [i in done_intervals | i[1]]
        SET f.`_dbcompleted_{part}_complete` = done_intervals = [[0, $_check_total_length-1]]
        SET f.`_dbcompleted_complete` = all(x in $_check_parts where f['_dbcompleted_'+x+'_complete'])
        """)
        completed = list(range(0, total_length)[slice])
        return q, {'_check_fname': fname, '_check_completed': completed,
                   '_check_parts': list(map(str, all_parts)), '_check_total_length': total_length}

    def file_batch_is_complete(self, filetype: File, path: Path, slc: slice, part: str) -> bool:
        """
        Returns True if the file[rows-slice][part] is complete otherwise False
        """
        total_length = filetype.length(self.rootdir / path, part)
        q = dedent(f"""
        OPTIONAL MATCH (f:File {{fname: $fname}})
        with f, apoc.coll.zip(f._dbcompleted_{part}_start, f._dbcompleted_{part}_end) as intervals
        return all(x in $rows where any(i in intervals where i[0] <= x and x <= i[1]))
        """)
        return self.graph.execute(q, fname=path.name, rows=list(range(total_length)[slc])).evaluate() or False

    def files_to_batches(self, *paths: Union[Path, str], batch_size=None, batches_slc=None, parts=None,
                         skip_complete=False):
        batches = []
        if not paths:
            return []
        if len(paths) == 1 and isinstance(paths[0], (tuple, list)):
            paths = paths[0]
        bar = tqdm(paths, desc='Building todo list')
        for path in bar:
            path = Path(path)
            matches = [f for f in self.filetypes if f.match_file(self.rootdir, path.relative_to(self.rootdir), self.graph)]
            if len(matches) > 1:
                raise ValueError(f"{path} matches more than 1 file type: {matches} with `{[m.match_pattern for m in matches]}`")
            filetype = matches[0]
            filetype_batch_size = filetype.recommended_batchsize if batch_size is None else batch_size
            slices = filetype.get_batches(path, filetype_batch_size, parts, batches_slc)
            bs = [(filetype, path.relative_to(self.rootdir), slc, part) for slc, part in slices]
            if skip_complete:
                bs = [b for b in bs if not self.file_batch_is_complete(*b)]
            batches += bs
        batches.sort(key=lambda b: (self.filetypes.index(b[0]), b[1]))  # make sure files are ingested in order of dependency
        return batches

    def write_files(self, *paths: Union[Path, str], raise_on_duplicate_file=False, skip_complete=True,
                    collision_manager='ignore', batch_size=None, parts=None, halt_on_error=True,
                    dryrun=False, do_not_apply_constraints=False, test_one=False, batches_slc: slice = None,
                    debug=False, debug_time=False, debug_params=False, debug_plan=False, timeout=None,
                    replan='skip_except_first', runtime='pipelined',
                    ) -> Optional[pd.DataFrame]:
        """
        Read in the files given in `paths` to the database.
        `collision_manager` is the method with which the database deals with overwriting data.
        Values of `collision_manager` can be {'ignore', 'overwrite', 'track&flag'}.
        track&flag will have the same behaviour as ignore but places the overlapping data in its own node for later retrieval.
        :return
            statistics dataframe
        """
        if not halt_on_error:
            logging.warning(f"`halt_on_error` is set to False. This would allow files to read without first reading their dependent files. "
                            f"This is not recommended.")
        if not do_not_apply_constraints:
            self.apply_constraints()
        batches = self.files_to_batches(*paths, batch_size=batch_size, batches_slc=batches_slc, parts=parts)
        if not batches:
            return
        elapsed_times = []
        stats = []
        timestamps = []
        if dryrun:
            logging.info(f"Dryrun: will not write to database. However, reading is permitted")
        _, path, slc, part = batches[0]
        desc_size = max(len(f'{path}[{slc.start}:{slc.stop}:{part}]') for filetype, path, slc, part in batches)
        first = f"{path}[{slc.start}:{slc.stop}:{part}]"
        bar = tqdm(batches, desc=first.ljust(desc_size))
        if debug or debug_time:
            with open('debug-timestamp.log', 'w') as f:
                pass
        out_batches = []
        for i, (filetype, path, slc, part) in enumerate(bar):
            bar.set_description(f'{path}[{slc.start}:{slc.stop}:{part}]')
            try:
                if raise_on_duplicate_file:
                    if len(self.graph.execute('MATCH (f:File {fname: $fname}) return count(f)', fname=path.name)) != 0:
                        raise FileExistsError(f"{path.name} exists in the DB and raise_on_duplicate_file=True")
                if skip_complete:
                    if self.file_batch_is_complete(filetype, path, slc, part):
                        continue
                with self.write_cypher(collision_manager) as query:
                    filetype.read(self.rootdir, path, slc, part)
                    cypher, params = query.render_query(as_lines=True)
                    uuid = f"//{uuid4()}"
                    guarantee, added_data = self.mark_batch_complete_query(path.name, slc, part, filetype.length(self.rootdir / path, part), filetype.parts)
                    if replan == 'skip_except_first':
                        replan_cypher = f'CYPHER replan=skip'
                        replan = 'skip'
                    else:
                        replan_cypher = f'CYPHER replan={replan}'
                    lines = [uuid, f'CYPHER runtime={runtime}', replan_cypher]  # runtime is to avoid neo4j bug with pipelines: https://github.com/neo4j/neo4j/issues/12441
                    if debug_plan:
                        lines.append('PROFILE')
                    lines += cypher[:-1] + ['WITH time0', guarantee, cypher[-1]]
                    params.update(added_data)
                    cypher = '\n'.join(lines)
                if debug:
                    with open('debug-query.log', 'w') as f:
                        f.write(cypher)
                    if debug_params:
                        with open('debug-params.log', 'w') as f:
                            f.write(self.graph.output_for_debug(**params, arrow=False, cmdline=False, silent=True))
                start = time.time()
                timed_out = False
                if not dryrun:
                    try:
                        results = self.graph.execute(cypher, **params)
                        stats.append(results.stats())
                        timestamp = results.evaluate()
                        successful = True
                    except (ConnectionError, RuntimeError, IndexError, ConnectionResetError) as e:
                        is_running = True
                        timed_out = False
                        while is_running:
                            is_running = self.graph.execute("CALL dbms.listQueries() YIELD query WHERE query STARTS WITH $uuid return count(*)", uuid=uuid).evaluate()
                            time.sleep(1)
                            bar.set_description(f'{path}[{slc.start}:{slc.stop}:{part}]')
                            if timeout is not None:
                                if time.time() - start > timeout:
                                    timed_out = True
                                    break
                        r = self.graph.execute('MATCH (f:File {fname: $fname}) return timestamp()', fname=path.name)
                        timestamp = r.evaluate()
                        successful = timestamp is not None
                        stats.append(r.stats())
                        if not successful and halt_on_error:
                            if timed_out:
                                raise ConnectionTimeOutError(f"Connection timed out whilst writing {path}{slc.start}:{slc.stop}:{part} and it failed") from e
                            else:
                                raise ConnectionError(f"{path} could not be written to the database see neo4j logs for more details") from e
                    if timestamp is None:
                        logging.warning(f"This query terminated early due to either an empty input table/data or "
                                        f"a match within the query returned no matches or it timed-out. "
                                        f"Adjust your `.read` method and query to allow for empty tables/data")
                    timestamps.append(timestamp)
                    if successful:
                        elapsed_times.append(time.time() - start)
                    else:
                        elapsed_times.append(-1)
                    out_batches.append((filetype, path, slc, part))
                    if debug or debug_time:
                        with open('debug-timestamp.log', 'a') as f:
                            f.write(str(elapsed_times[-1]) + '\n')
            except (ClientError, DatabaseError, FileExistsError, ConnectionError) as e:
                logging.exception('ClientError:', exc_info=True)
                if halt_on_error:
                    raise e
                print(e)
            if test_one:
                batches = batches[i:i+1]
                logging.info(f"Writing terminated because `test_one=True`")
                break
        if len(out_batches) and not dryrun:
            df = pd.DataFrame(stats)
            df['timestamp'] = timestamps
            df['elapsed_time'] = elapsed_times
            _, df['fname'], slcs, parts = zip(*out_batches)
            df['batch_start'], df['batch_end'] = zip(*[(i.start, i.stop) for i in slcs])
            df['part'] = parts
        elif dryrun:
            pass
            # df = pd.DataFrame(columns=['elapsed_time', 'fname', 'batch_start', 'batch_end', 'part'])
            # df['elapsed_time'] = elapsed_times
        else:
            df = pd.DataFrame(columns=['timestamp', 'elapsed_time', 'fname', 'batch_start', 'batch_end'])
        if dryrun:
            return
        return df.set_index(['fname', 'batch_start', 'batch_end', 'part'])

    def find_files(self, *filetype_names, skip_complete_files=True):
        filetype_names = [x+'_file' if not x.endswith('_file') else x for x in filetype_names]
        if any(f for f in filetype_names if f not in [i.singular_name for i in self.filetypes]):
            raise KeyError(f"Some or all of the filetype_names are not understood. "
                           f"Allowed names are: {[i.singular_name.replace('_file', '') for i in self.filetypes]}")
        filelist = []
        if len(filetype_names) == 0:
            filetypes = self.filetypes
        else:
            filetypes = [f for f in self.filetypes if f.singular_name in filetype_names]

        for filetype in filetypes:
            filelist += sorted([i for i in filetype.match_files(self.rootdir, self.graph)], key=lambda f: f.name)
        if skip_complete_files:
            extant_fnames = self.get_complete_files() if skip_complete_files else []
            filtered_filelist = [i for i in filelist if str(i.name) not in extant_fnames]
        else:
            filtered_filelist = filelist
        diff = len(filelist) - len(filtered_filelist)
        if diff:
            print(f'Skipping {diff} complete files (use skip_complete_files=False to go over them again)')
        return [f for f in tqdm(filtered_filelist, desc='getting MOS files') if File.check_mos(f)]

    def write_directory(self, *filetype_names, collision_manager='ignore', skip_extant_files=True, halt_on_error=False,
                        batch_size=None, parts=None, dryrun=False, **kwargs) -> pd.DataFrame:
        filtered_filelist = self.find_files(*filetype_names, skip_extant_files=skip_extant_files)
        return self.write_files(*filtered_filelist, collision_manager=collision_manager, halt_on_error=halt_on_error,
                                dryrun=dryrun, batch_size=batch_size, parts=parts, **kwargs)

    def _validate_one_required(self, hierarchy_name, timestamp=None):
        hierarchy = self.singular_hierarchies[hierarchy_name]
        if hierarchy.is_template:
            return
        subclasses = get_all_subclasses(hierarchy)
        parents = [h for h in hierarchy.parents]
        children = [h for h in hierarchy.children]
        dfs = []
        for relation in parents+children:
            if isinstance(relation, Multiple):
                mn, mx = relation.minnumber, relation.maxnumber
                b = relation.node.__name__
                reflected = relation.one2one
                relation_idname = relation.relation_idname
            else:
                mn, mx = 1, 1
                b = relation.__name__
                reflected = False
                relation_idname = None
            mn = 0 if mn is None else mn
            mx = 9999999 if mx is None else mx
            a = hierarchy.__name__
            if relation in parents:
                arrows = ['<-', '-']
                reldef = [a, b]
            else:
                arrows = ['-', '->']
                reldef = [b, a]
            relid1 = f'WHERE r1.relation_id = "{relation_idname}"' if relation_idname else ''
            relid2 = f'WHERE r2.relation_id = "{relation_idname}"' if relation_idname else ''
            subclass_exclusions = ' WHERE ' + ' AND '.join([f'NOT n:{i.__name__}' for i in subclasses]) if len(subclasses) else ''
            q = f"""
            MATCH (n:{a}){subclass_exclusions}
            OPTIONAL MATCH (n){arrows[0]}[r1:is_required_by]{arrows[1]}(m:{b}) {relid1}
            WITH n, count(r1) as forwardCount
            OPTIONAL MATCH (m:{b}){arrows[0]}[r2:is_required_by]{arrows[1]}(n) {relid2}
            WITH n, forwardCount, count(r2) as reflCount           
            WITH *
            WHERE forwardCount < {mn} OR forwardCount > {mx} OR (reflCount <> forwardCount AND {reflected})
            RETURN "{reldef[0]}" as child, "{reldef[1]}" as parent, {mn} as mn, {mx} as mx, {reflected} as reflected, id(n), forwardCount, reflCount            
            """
            result = self.graph.neograph.run(q).to_data_frame()
            dfs.append(result)
        try:
            df = pd.concat(dfs)
            return df, df.groupby(['child', 'parent']).apply(len).astype(int)
        except KeyError:
            return df, pd.Series()
        except ValueError:
            return pd.DataFrame(), pd.Series()

    def _validate_no_duplicate_relation_ordering(self, time=None):
        q = """
        MATCH (a)-[r1]->(b)<-[r2]-(a)
        WHERE TYPE(r1) = TYPE(r2) AND r1.order <> r2.order
        WITH a, b, apoc.coll.union(COLLECT(r1), COLLECT(r2))[1..] AS rs
        RETURN DISTINCT labels(a), id(a), labels(b), id(b), count(rs)+1
        """
        return self.graph.neograph.run(q).to_data_frame()

    def _validate_no_duplicate_relationships(self, timestamp=None):
        q = """
        MATCH (a)-[r1]->(b)<-[r2]-(a)
        WHERE TYPE(r1) = TYPE(r2) AND PROPERTIES(r1) = PROPERTIES(r2)
        WITH a, b, apoc.coll.union(COLLECT(r1), COLLECT(r2))[1..] AS rs
        RETURN DISTINCT labels(a), id(a), labels(b), id(b), count(rs)+1
        """
        return self.graph.neograph.run(q).to_data_frame()

    def _validate_rel_unique_ids(self, *hiers):
        """
        For each hierarchy which defines an identifier builder based on parent hierarchies,
        check that the parents+self is unique in the db since this is not guaranteed by neo4j indexes
        """
        instance_duplications = []
        for h in tqdm(hiers or list(self.singular_hierarchies.keys()), desc='checking for schema uniqueness violations'):
            hierarchy = self.singular_hierarchies[h]
            if hierarchy.has_rel_identity() and not hierarchy.is_template:
                id_factors = [f for f in hierarchy.factors if f in hierarchy.identifier_builder]
                id_parents = [p if isinstance(p, Multiple) else OneOf(p) for p in hierarchy.parents]
                id_children = [p if isinstance(p, Multiple) else OneOf(p) for p in hierarchy.children]
                id_parents = {p.node.__name__: (p.minnumber, p.maxnumber) for p in id_parents if p.name in hierarchy.identifier_builder}
                id_children = {p.node.__name__: (p.minnumber, p.maxnumber) for p in id_children if p.name in hierarchy.identifier_builder}
                parent_nodes = [f"<-[rp{i}:is_required_by]-(p{i}:{p})" for i, p in enumerate(id_parents)]
                parent_nodes2 = [f"<-[rp{i}2:is_required_by]-(p{i})" for i, p in enumerate(id_parents)]
                child_nodes = [f"-[rc{i}:is_required_by]->(c{i}:{c})" for i, c in enumerate(id_children)]
                child_nodes2 = [f"-[rc{i}2:is_required_by]->(c{i})" for i, c in enumerate(id_children)]
                nodes = parent_nodes+child_nodes
                nodes2 = parent_nodes2+child_nodes2
                node = f"(n:{hierarchy.__name__})"
                node2 = f"(m:{hierarchy.__name__})"
                first = f"{node}{nodes[0]}"
                first2 = f"{node2}{nodes2[0]}"
                others = ','.join([f"(n){p}" for p in nodes[1:]])
                others2 = ','.join([f"(m){p}" for p in nodes2[1:]])
                parent_vars = ','.join([f"p{i}" for i in range(len(parent_nodes))]+[f"c{i}" for i in range(len(child_nodes))])
                rel_vars = ','.join([f"rp{i}.order" for i in range(len(parent_nodes))]+[f"rc{i}.order" for i in range(len(child_nodes))])
                rel_vars2 = ','.join([f"rp{i}2.order" for i in range(len(parent_nodes2))]+[f"rc{i}2.order" for i in range(len(child_nodes2))])
                property_eq = ' and '.join([f"m.{f} = n.{f}" for f in id_factors])
                if property_eq:
                    property_eq = f' and {property_eq}'
                if others:
                    first = f"{first},{others}"
                if others2:
                    first2 = f"{first2},{others2}"
                q = f"""
                MATCH {first}
                WITH n, {parent_vars}, [{rel_vars}] as coll1
                OPTIONAL MATCH {first2}
                where n <> m {property_eq} and [{rel_vars2}] = coll1
                WITH n, count(m) as nduplicates, count(*) as cnt_rels
                where nduplicates = cnt_rels  // if even 1 is different, then it's not a duplicate
                return labels(n)[0] as type, id(n)
                """
                duplicates = self.graph.neograph.run(q).to_data_frame()
                if len(duplicates):
                    instance_duplications.append(duplicates)
        try:
            instance_duplications = pd.concat(instance_duplications)
            class_duplications = instance_duplications.groupby('type').apply(len)
            return instance_duplications, class_duplications
        except ValueError:
            return [], []


    def _validate_expected_number_schema(self):
        schema_violations = []
        label_instances = []
        for h in tqdm(list(self.singular_hierarchies.keys()), desc='checking for schema parent/child # violations'):
            df = self._validate_one_required(h)
            if df is not None:
                df, label_instance = df
                schema_violations.append(df)
                label_instances.append(label_instance)
        schema_violations = pd.concat(schema_violations)
        label_instances = pd.concat(label_instances)
        return schema_violations, label_instances

    def _validate_relation_constraints(self):
        qs = []
        for h in tqdm(list(self.singular_hierarchies.keys()), desc='checking for rel constraint schema violations'):
            hierarchy = self.singular_hierarchies[h]
            if not hierarchy.is_template:
                for rel in hierarchy.parents + hierarchy.children:
                    if isinstance(rel, Multiple):
                        for constraint in rel.constrain:
                            q = f"""
                            MATCH (h:{hierarchy.__name__})--(r:{rel.node.__name__})--(c:{constraint.__name__})
                            WITH h, r, size(collect(c)) as size
                            where size <> 1
                            return h as node, r as dependent, "{constraint.__name__}" as constraint, size as ndifferent_constaints
                            """
                            qs.append(self.graph.neograph.run(q).to_data_frame())
        return qs

    def _validate_all_nodes_created_with_file(self):
        q = dedent("""match (f:File) 
        with collect(distinct f._dbcreated) as dbs
        match (n) where not n.`_dbcreated` in dbs
        with labels(n) as labels, count(n) as cnt, collect(id(n)) as ids
        RETURN labels, cnt, ids""")
        return self.graph.execute(q).to_data_frame()

    def validate(self, timestamp: int = None, file_paths: List[Path]=None):
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.INFO)
        missing = []
        with pd.option_context('display.max_columns', None):
            if file_paths is not None:
                logging.info(f"Checking all requested files are present...")
                extant = {path: fname for path, fname in self.graph.execute(f'MATCH (n:File) return n.path as path, n.fname as fname')}
                missing = [str(path) for path in file_paths if path in extant]
                if missing:
                    logging.warning('fail')
                    logging.warning(f"\tMissing {len(missing)} files from db")
                    logging.warning(f"\tMissing the following files in the database:")
                    logging.warning(f"{missing}")
                else:
                    logging.info('pass')
            logging.info(f"Checking no duplicate relationships...")
            duplicates = self._validate_no_duplicate_relationships(timestamp)
            if len(duplicates):
                logging.warning(f"fail")
                logging.warning(f'\tThere are {len(duplicates)} duplicate relations')
                logging.warning(f"\n{duplicates}")
                logging.warning(f"\tThis indicates a weaveio logic issue. Raise with maintainer.")
            else:
                logging.info(f"pass")
            logging.info(f"Checking no duplicate relationships with different ordering...")
            duplicates = self._validate_no_duplicate_relation_ordering(timestamp)
            if len(duplicates):
                logging.warning(f"fail")
                logging.warning(f'\tThere are {len(duplicates)} relations with different orderings')
                logging.warning(f"\n{duplicates}")
                logging.warning(f"\tThis indicates a weaveio logic issue. Raise with maintainer.")
            else:
                logging.info(f"pass")
            logging.info(f"Checking expected number of parent/child relations from schema...")
            schema_violations, label_instances = self._validate_expected_number_schema()
            if len(schema_violations):
                logging.warning(f"fail")
                logging.warning(f'\tThere are {len(schema_violations)} violations of expected relationship number ({len(label_instances)} class relations)')
                logging.warning(f"\n{schema_violations}")
                logging.warning(f"\n{label_instances}")
                logging.warning(f"\tThis indicates incorrect file read logic on user defined read functions. Check read function.")
            else:
                logging.info(f"pass")
            logging.info(f"Checking unique relationships based on connected nodes are unique...")
            unique_rel_id_duplicates, label_rel_id_duplicates = self._validate_rel_unique_ids()
            if len(unique_rel_id_duplicates):
                logging.warning(f"fail")
                logging.warning(f"\tThere are {len(unique_rel_id_duplicates)} unique rel id violations ({len(label_rel_id_duplicates)} class definitions)")
                logging.warning(f"\n{unique_rel_id_duplicates}")
                logging.warning(f"\n{label_rel_id_duplicates}")
                logging.warning(f"\tThis indicates a weaveio logic issue. Raise with maintainer.")
            else:
                logging.info(f"pass")
            if len(missing) + len(duplicates) + len(schema_violations) + len(unique_rel_id_duplicates) == 0:
                logging.info('Database schema is valid!')
            else:
                logging.warning(f"Database schema is invalid")
            logging.getLogger().setLevel(level)
            return duplicates, (schema_violations, label_instances), unique_rel_id_duplicates

    def is_product(self, factor_name, hierarchy_name):
        return self.singular_name(factor_name) in self.singular_hierarchies[self.singular_name(hierarchy_name)].products

    def is_factor_name(self, name):
        if name in self.factor_hierarchies:
            return True
        try:
            name = self.singular_name(name)
            return self.is_singular_factor(name) or self.is_singular_idname(name)
        except KeyError:
            return False

    def is_singular_idname(self, value):
        return self.is_singular_name(value) and value.split('.')[-1] in self.singular_idnames

    def is_plural_idname(self, value):
        return self.is_plural_name(value) and value.split('.')[-1] in self.plural_idnames

    def is_plural_factor(self, value):
        return self.is_plural_name(value) and value.split('.')[-1] in self.plural_factors

    def is_singular_factor(self, value):
        return self.is_singular_name(value) and value.split('.')[-1] in self.singular_factors

    def class_name(self, name):
        if isinstance(name, type):
            return name.__name__
        else:
            return self.singular_hierarchies[self.singular_name(name)].__name__

    def plural_name(self, name):
        if isinstance(name, type):
            if issubclass(name, Hierarchy):
                return name.plural_name
            else:
                raise TypeError(f'{name} is not a weaveio object or string')
        if name in self.class_hierarchies:
            return self.class_hierarchies[name].plural_name
        name = name.lower()
        if self.is_plural_name(name):
            return name
        if self.is_singular_name(name):
            try:
                return self.singular_factors[name]
            except KeyError:
                try:
                    return self.singular_idnames[name]
                except KeyError:
                    try:
                        return self.relative_names[name]
                    except KeyError:
                        return self.singular_hierarchies[name].plural_name
        if '.' in name:
            pattern = name.lower().split('.')
            if any(map(self.is_plural_name, pattern)):
                return name
            return '.'.join(pattern[:-1] + [self.plural_name(pattern[-1])])
        return make_plural(name)

    def singular_name(self, name):
        if isinstance(name, type):
            if issubclass(name, Hierarchy):
                return name.singular_name
            else:
                raise TypeError(f'{name} is not a weaveio object or string')
        if name in self.class_hierarchies:
            return self.class_hierarchies[name].singular_name
        name = name.lower()
        if self.is_singular_name(name):
            return name
        if self.is_plural_name(name):
            try:
                return self.plural_factors[name]
            except KeyError:
                try:
                    return self.plural_idnames[name]
                except KeyError:
                    try:
                        return self.plural_relative_names[name]
                    except KeyError:
                        return self.plural_hierarchies[name].singular_name
        if '.' in name:
            pattern = name.lower().split('.')
            return '.'.join([self.singular_name(p) for p in pattern])
        return make_singular(name)

    def is_valid_name(self, name):
        if not isinstance(name, str):
            return False
        name = name.lower()
        if self.is_plural_name(name) or self.is_singular_name(name):
            return True
        if isinstance(name, str):
            pattern = name.split('.')
            if len(pattern) == 1:
                return self.is_plural_name(name) or self.is_singular_name(name)
            return all(self.is_valid_name(p) for p in pattern)
        return False

    def is_plural_name(self, name):
        """
        Returns True if name is a plural name of a hierarchy
        e.g. spectra is plural for Spectrum
        """
        if not isinstance(name, str):
            return False
        name = name.lower()
        if name in self.plural_hierarchies or name in self.plural_factors or\
                   name in self.plural_idnames or name in self.plural_relative_names:
            return True
        pattern = name.split('.')
        if len(pattern) == 1:
            return name in self.plural_hierarchies or name in self.plural_factors or\
                   name in self.plural_idnames or name in self.plural_relative_names
        return all(self.is_plural_name(n) for n in pattern)

    def is_singular_name(self, name):
        if not isinstance(name, str):
            return False
        name = name.lower()
        if name in self.singular_hierarchies or name in self.singular_factors or \
           name in self.singular_idnames or name in self.relative_names:
            return True
        pattern = name.split('.')
        if len(pattern) == 1:
            return name in self.singular_hierarchies or name in self.singular_factors or \
                   name in self.singular_idnames or name in self.relative_names
        return all(self.is_singular_name(n) for n in pattern)

    def __getitem__(self, address):
        if not isinstance(address, str):
            raise KeyError(f"{self} cannot be indexed by integers.")
        try:
            return self.query.__getitem__(address)
        except (UserError, KeyError) as e:
            self.query.raise_error_with_suggestions(address, e)

    def __getattr__(self, item):
        if item in ['keys', 'items', 'values']:
            # catch fastapi jsonable_encoder requests for serialization
            raise AttributeError(f"{self} has no attribute {item}")
        if item.startswith('_'):
            # query attributes are not allowed to start with '_'
            # if we are here, then that means __getattribute__ has failed and we should fail here
            # like a normal object
            raise AttributeError(f"{self} has no attribute {item}")
        try:
            return self.query.__getattr__(item)
        except (UserError, KeyError) as e:
            self.query.raise_error_with_suggestions(item, e)

    def plot_relations(self, i=-1, show_hdus=True, fname='relations', format='pdf', include=None):
        graph = self.relation_graphs[i]
        if not show_hdus:
            graph = nx.subgraph_view(graph, lambda n: not issubclass(n, HDU))  # True to get rid of templated
        # G = nx.subgraph_view(graph, filter_edge=lambda a, b: graph.edges[a, b]['style'] !=  'dotted')
        G = graph
        if include is not None:
            include = [self.singular_hierarchies[i] for i in include]
            include_list = include.copy()
            include_list += [a for i in include for a in nx.ancestors(G, i)]
            include_list += [d for i in include for d in nx.descendants(G, i)]
            G = nx.subgraph_view(G, lambda n: n in include_list)
        plot_graph(G, fname, format)

    def find_names(self, guess, n=10):
        """
        Finds the closest name to guess in the hierarchy. Returns the top n matches.
        """
        sources = [self.singular_hierarchies, self.plural_hierarchies, self.singular_factors, self.plural_factors]
        suggestions = {i for source in sources for i in source}
        inorder = sorted(suggestions, key=lambda x: textdistance.jaro_winkler(guess, x), reverse=True)
        return inorder[0:n]


    def _autosuggest(self, a, relative_to=None):
        a = self.singular_name(a)
        if relative_to is not None:
            relative_to = self.singular_name(relative_to)
        distance, distance_reverse = textdistance.jaro_winkler, True
        suggestions = []

        for h_singular_name, h in self.singular_hierarchies.items():
            newsuggestions = []
            if relative_to is not None:
                try:
                    self.paths_to_hierarchy(relative_to, h_singular_name, singular=True)
                except NetworkXNoPath:
                    hier = h.singular_name
                    factors = h.products_and_factors
                except NodeNotFound:
                    continue
                else:
                    hier = h.plural_name
                    factors = [self.plural_name(f) for f in h.products_and_factors]
                newsuggestions.append(hier)
                newsuggestions += factors
            else:
                newsuggestions += [h.singular_name, h.plural_name] + h.products_and_factors
                newsuggestions += [self.plural_name(f) for f in h.products_and_factors]
            try:
                newsuggestions.index(a)
            except ValueError:
                suggestions += newsuggestions
            else:
                return [a]
        inorder = sorted(list(set(suggestions)), key=lambda x: distance(a, x), reverse=distance_reverse)
        return inorder[:3]

    def autosuggest(self, a: str, relative_to: str = None, exception: Exception = None):
        if isinstance(exception, CardinalityError):
            l = [self.plural_name(a)]
        else:
            l = self._autosuggest(a, relative_to)
        string = '\n'.join([f'{i}. {s}' for i, s in enumerate(l, start=1)])
        msg = f"did you mean one of:\n{string}"
        raise exception.__class__(f"{str(exception.args[0])}\n{msg}") from exception

    def _get_by_neo4j_id(self, neoid, plural_name):
        from weaveio.readquery.functions import neo4j_id
        q = self.query.__getattr__(plural_name)
        return q[neo4j_id(q) == int(neoid)]