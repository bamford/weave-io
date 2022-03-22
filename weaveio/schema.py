import logging
from collections import deque
from typing import Type, List, Union, Tuple, Dict, Set

import py2neo
from py2neo import Relationship, Node, Subgraph
from py2neo.cypher import Cursor
from py2neo.export import Table
from tqdm import tqdm

from weaveio.graph import Graph
from weaveio.hierarchy import Graphable, Hierarchy, Multiple, One2One, Optional


class AttemptedSchemaViolation(Exception):
    pass


def get_all_class_bases(cls: Type[Graphable]) -> List[Type[Graphable]]:
    new = []
    for b in cls.__bases__:
        if b is Graphable or not issubclass(b, Graphable):
            continue
        new.append(b)
        new += get_all_class_bases(b)
    return new


def get_labels_of_schema_hierarchy(hierarchy: Union[Type[Hierarchy], Multiple], as_set=False):
    if isinstance(hierarchy, Multiple):
        hierarchy = hierarchy.node
    bases = get_all_class_bases(hierarchy)
    labels = [i.__name__ for i in bases]
    labels.append(hierarchy.__name__)
    labels.append('SchemaNode')
    if as_set:
        return frozenset(labels)
    return labels


def push_py2neo_schema_subgraph_cypher(subgraph: Subgraph) -> Tuple[str, Dict]:
    """
    Each node label is unique, so we just merge by label and then update properties
    for relationships we merge with empty relation and then update the properties
    """
    cypher = []
    params = {}
    for i, node in enumerate(subgraph.nodes):
        cypher.append(f"MERGE (n{i}{node.labels}) WITH * WHERE size(labels(n{i})) = {len(node.labels)}")
        for k, v in node.items():
            params[f'n{i}{k}'] = v
            cypher.append(f"SET n{i}.{k} = $n{i}{k}")
    for i, rel in enumerate(subgraph.relationships):
        a = list(subgraph.nodes).index(rel.start_node)
        b = list(subgraph.nodes).index(rel.end_node)
        rel_params = []
        for k, v in rel.items():
            params[f'r{i}{k}'] = v
            rel_params.append(f"{k}: $r{i}{k}")
        rel_params = ','.join(rel_params)
        cypher.append(f"MERGE (n{a})-[r{i}:{list(rel.types())[0]} {{{rel_params}}}]->(n{b})")
    return "\n".join(cypher), params

def int_or_none(x):
    if x is None:
        return None
    return int(x)

def get_all_schema_nodes(graph):
    return graph.execute("MATCH (n:SchemaNode) return n.name, labels(n), properties(n)").to_table()

def get_all_schema_rows(graph, hierarchy_name) -> Table:
    cypher = f"""
    MATCH (n:{hierarchy_name}:SchemaNode {{name: '{hierarchy_name}'}})
    OPTIONAL MATCH (n)-[child_rel:HAS_CHILD]->(child:SchemaNode) WHERE child_rel.originated = n.name
    with n, child, collect(child_rel) as child_rels
    OPTIONAL MATCH (parent:SchemaNode)-[parent_rel:IS_PARENT_OF]->(n) WHERE parent_rel.originated = n.name
    with n, child, child_rels, parent, collect(parent_rel) as parent_rels
    RETURN n, parent, child, parent_rels, child_rels
    """
    return graph.execute(cypher).to_table()

def get_type_hierarchy_subgraph(graph, hierarchy):
    get_all_class_bases()

def diff_hierarchy_schema_node(graph: Graph, hierarchy: Type[Hierarchy]):
    """
    Given a hierarchy, return py2neo subgraph that can be pushed to update the schema.
    If the new hierarchy is not backwards compatible, an exception will be raised

    A hierarchy can only be merged into the schema if existing data will still match the schema.
    i.e. if all below are true:
        has the same idname
        has the same factors or more
        has the same parents and children
            (additional parents and children can only be specified if they are optional)

    In the schema:
        (a)<-[:has_parent]-(b) indicates that (b) requires a parent (a) at instantiation time
        (a)-[:has_child]->(b) indicates that (a) requires a child (b) at instantiation time
    """
    # structure is [{labels}, is_optional, (minn, maxn), rel_idname, is_one2one, class_name]
    actual_parents = [(get_labels_of_schema_hierarchy(p, True), False, (1, 1), None, False, p.__name__) if not isinstance(p, Multiple)
                      else (get_labels_of_schema_hierarchy(p.node, True), p.minnumber == 0,
                            (p.minnumber, p.maxnumber), p.relation_idname, isinstance(p, One2One), p.node.__name__) for p in hierarchy.parents]
    actual_children = [(get_labels_of_schema_hierarchy(p, True), False, (1, 1), None, False, p.__name__) if not isinstance(p, Multiple)
                       else (get_labels_of_schema_hierarchy(p.node, True), p.minnumber == 0,
                             (p.minnumber, p.maxnumber), p.relation_idname, isinstance(p, One2One), p.node.__name__) for p in hierarchy.children]
    results = get_all_schema_rows(graph, hierarchy.__name__)
    labels = get_labels_of_schema_hierarchy(hierarchy)
    if len(results) == 0:  # node is completely new
        parents = []
        children = []
        found_node = Node(*labels,
                          factors=hierarchy.factors, idname=hierarchy.idname,
                          name=hierarchy.__name__,
                          singular_name=hierarchy.singular_name,
                          plural_name=hierarchy.plural_name)
        for struct in actual_parents:
            props = dict(optional=struct[1], minnumber=struct[2][0], maxnumber=struct[2][1],
                         idname=struct[3], one2one=struct[4])
            parent = Node(*struct[0], name=struct[5])  # extant parent, specify labels so it matches not creates
            parents.append([parent, props])
        for struct in actual_children:
            props = dict(optional=struct[1], minnumber=struct[2][0], maxnumber=struct[2][1],
                         idname=struct[3], one2one=struct[4])
            child = Node(*struct[0], name=struct[5])  # extant child, specify labels so it matches not creates
            children.append([child, props])
    else:
        found_node = results[0][0]
        actual_parents = set(actual_parents)
        actual_children = set(actual_children)

        # gather extant info
        found_factors = set(found_node.get('factors', []))
        found_parents = {(frozenset(r[1].labels), bool(rel['optional']),
                          (int_or_none(rel['minnumber']), int_or_none(rel['maxnumber'])),
                          rel['idname'], bool(rel['one2one']), r[1]['name']) for r in results if r[1] is not None for rel in r[3]}
        found_children = {(frozenset(r[2].labels), bool(rel['optional']),
                           (int_or_none(rel['minnumber']), int_or_none(rel['maxnumber'])),
                           rel['idname'], bool(rel['one2one']), r[2]['name']) for r in results if r[2] is not None for rel in r[4]}

        # see if hierarchy is different in any way
        different_idname = found_node.get('idname') != hierarchy.idname
        different_singular_name = found_node.get('singular_name') != hierarchy.singular_name
        different_plural_name = found_node.get('plural_name') != hierarchy.plural_name
        missing_factors = set(hierarchy.factors) - found_factors
        different_labels = set(labels).symmetric_difference(found_node.labels)

        # parents are different?
        missing_parents = set(actual_parents - found_parents)
        new_parents = {p for p in found_parents - actual_parents if not p[1]}  # allowed if the new ones are optional

        # children are different?
        missing_children = found_children - actual_children  # missing from new definition
        new_children = {p for p in actual_children - found_children if not p[1]}  # allowed if the new ones are optional

        if different_idname or missing_factors or different_labels or missing_parents or \
                new_parents or missing_children or new_children or different_singular_name or different_plural_name:
            msg = f'Cannot add new hierarchy {hierarchy} because the {hierarchy.__name__} already exists' \
                  f' and the proposed definition of {hierarchy} is not backwards compatible. ' \
                  f'The differences are listed below:\n'
            if different_idname:
                msg += f'- proposed idname {hierarchy.idname} is different from the original {found_node.get("idname")}\n'
            if different_singular_name:
                msg += f'- proposed singular_name {hierarchy.singular_name} is different from the original {found_node.get("singular_name")}\n'
            if different_plural_name:
                msg += f'- proposed plural_name {hierarchy.plural_name} is different from the original {found_node.get("plural_name")}\n'
            if different_labels:
                msg += f'- proposed inherited types {different_labels} are different from {labels}\n'
            if missing_factors:
                msg += f'- factors {missing_factors} are missing from proposed definition\n'
            if new_parents:
                msg += f'- new parents with labels {[set(p[0]) - {"SchemaNode"} for p in new_parents]} are not optional (and therefore arent backwards compatible)\n'
            if new_children:
                msg += f'- new children with labels {[set(p[0]) - {"SchemaNode"} for p in new_children]} are not optional (and therefore arent backwards compatible)\n'
            if missing_parents:
                msg += f'- parents with labels {[set(p[0]) - {"SchemaNode"} for p in missing_parents]} are missing from the new definition\n'
            if missing_children:
                msg += f'- children with labels {[set(p[0]) - {"SchemaNode"} for p in missing_children]} are missing from the new definition\n'
            msg += f'any flagged children or parents may have inconsistent min/max number'
            raise AttemptedSchemaViolation(msg)

        nodes = []
        if set(hierarchy.factors) != found_factors:
            found_node['factors'] = hierarchy.factors  # update
            nodes.append(found_node)
        if found_children.symmetric_difference(actual_children):
            children = [(Node(*labels, name=name), dict(optional=optional, minnumber=minn, maxnumber=maxn,
                                             idname=idname, one2one=one2one))
                        for labels, optional, (minn, maxn), idname, one2one, name in actual_children]
            nodes += children
        else:
            children = []
        if found_parents.symmetric_difference(actual_parents):
            parents = [(Node(*labels, name=name), dict(optional=optional, minnumber=minn, maxnumber=maxn,
                                            idname=idname, one2one=one2one))
                       for labels, optional, (minn, maxn), idname, one2one, name in actual_parents]
            nodes += parents
        else:
            parents = []
    rels = []
    for c, props in children:
            rels.append(Relationship(found_node, 'HAS_CHILD', c, **props, originated=found_node.get('name')))
            if props['one2one']: # reflect the relationship if it is One2One
                rels.append(Relationship(c, 'HAS_CHILD', found_node, **props, originated=found_node.get('name')))
    for p, props in parents:
            rels.append(Relationship(p, 'IS_PARENT_OF', found_node, **props, originated=found_node.get('name')))
            if props['one2one']: # reflect the relationship if it is One2One
                rels.append(Relationship(found_node, 'IS_PARENT_OF', p, **props, originated=found_node.get('name')))
    return Subgraph([i[0] for i in parents + children] + [found_node], rels)


def write_schema(graph, hierarchies, dryrun=False, return_subgraph=False):
    """
    writes to the neo4j schema graph for use in optimising queries
    this should always be done before writing data
    """
    # sort available hierarchies meaning non-dependents first
    hierarchies = deque(sorted(list(hierarchies),
                               key=lambda x: len(x.children) + len(x.parents) + len(x.products)))
    done = []
    executions = []
    nmisses = 0
    while hierarchies:
        if nmisses == len(hierarchies):
            raise AttemptedSchemaViolation(f"Dependency resolution impossible, proposed schema elements may have cyclic dependencies:"
                                           f"{list(hierarchies)}")
        hier = hierarchies.popleft()  # type: Type[Hierarchy]
        hier.instantate_nodes()
        if hier.is_template:
            continue  # don't need to add templates, they just provide labels
        dependencies = [d.node if isinstance(d, Multiple) else d for d in hier.parents + hier.children]
        dependencies = filter(lambda d: d != hier, dependencies)  # allow self-references
        if not all(d in done for d in dependencies):
            logging.info(f"{hier} put at the back because it requires dependencies which have not been written yet")
            hierarchies.append(hier)  # do it after the dependencies are done
            nmisses += 1
            continue
        subgraph = diff_hierarchy_schema_node(graph, hier)
        executions.append(push_py2neo_schema_subgraph_cypher(subgraph))
        # executions.append(push_py2neo_schema_subgraph_cypher(get_type_hierarchy_subgraph(graph, hier))
        nmisses = 0  # reset counter
        done.append(hier)
    if not dryrun:
        for cypher, params in tqdm(executions, desc='schema updates'):
            graph.execute(cypher, **params)
    return True


def get_dependencies_from_schema_rows(rows):
    parents = {row[1].get('name'): row[3][0] for row in rows if row[3]}
    children = {row[2].get('name'): row[4][0] for row in rows if row[4]}
    return parents, children


def parse_dependency(hierarchies: Dict[str, Type[Hierarchy]], dependency: Relationship) -> Union[Multiple, Type[Hierarchy]]:
    hierarchy = hierarchies[dependency.start_node['name']]
    mn, mx = dependency.get('minnumber', None), dependency.get('minnumber', None)
    if dependency['one2one']:
        dep = One2One(hierarchy, idname=dependency['name'])
    elif dependency['optional']:
        dep = Optional(hierarchy, idname=dependency['name'])
    elif mn is not None or mx is not None:
        dep = Multiple(hierarchy, mn, mx, idname=dependency['name'])
    else:
        dep = hierarchy
    return dep


class SchemaNode:
    pass


def read_schema(graph) -> Set[Type[Hierarchy]]:
    bases_and_attrs = {r[0]: (r[1], r[2]) for r in get_all_schema_nodes(graph)}
    relation_tables = [(name, get_dependencies_from_schema_rows(get_all_schema_rows(graph, name))) for name, (bases, attrs) in bases_and_attrs.items()]
    relation_tables = deque(sorted(relation_tables, key=lambda x: len(x[1][0]) + len(x[1][1])))
    done = {'Hierarchy': Hierarchy, 'SchemaNode': SchemaNode}
    nmisses = 0
    while relation_tables:
        if nmisses == len(relation_tables):
            raise AttemptedSchemaViolation(f"Dependency resolution impossible, read schema elements may have cyclic dependencies:"
                                           f"{[i[0] for i in relation_tables]}")
        name, (parents, children) = relation_tables.popleft()
        if not all(parent in done for parent in parents) or not all(child in done for child in children):
            logging.info(f"{name} put at the back because it requires dependencies which have not been written yet")
            relation_tables.append((name, (parents, children)))  # do it after the dependencies are done
            nmisses += 1
            continue
        base_names, attrs = bases_and_attrs[name]
        bases = [done.get(base) for base in base_names]
        parents = [parse_dependency(done, parent) for parent in parents.values()]
        children = [parse_dependency(done, child) for child in children.values()]
        attrs['parents'] = parents
        attrs['children'] = children
        done[name] = type(name, bases, attrs)
    for k, v in done.items():
        v.instantate_nodes()
    return set(done.values())


# TODO: put templates in and also label class dependency tree in neo4j
