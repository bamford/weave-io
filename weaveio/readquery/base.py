from typing import Tuple, List, Dict, Any, Union, TYPE_CHECKING

from .parser import QueryGraph

if TYPE_CHECKING:
    from .objects import ObjectQuery, Query, AttributeQuery
    from ..data import Data


class AmbiguousPathError(Exception):
    pass


class CardinalityError(Exception):
    pass


class BaseQuery:
    one_row = False
    one_column = False

    def __repr__(self):
        return f'<{self.__class__.__name__}({self._previous._obj}-{self._obj})>'

    def _compile(self) -> Tuple[List[str], Dict[str, Any], List[str]]:
        """
        returns the cypher lines, cypher parameters, names of columns, expect_one_row, expect_one_column
        """
        return self._G.cypher_lines(self._node), self._G.parameters, self._names

    def __init__(self, data: 'Data', G: QueryGraph = None, node=None, previous: Union['Query', 'AttributeQuery', 'ObjectQuery'] = None,
                 obj: str = None, start: 'Query' = None, index_node = None,
                 single=False, names=None, *args, **kwargs) -> None:
        from .objects import ObjectQuery
        self._single = single
        self._data = data
        if G is None:
            self._G = QueryGraph()
        else:
            self._G = G
        if node is None:
            self._node = self._G.start
        else:
            self._node = node
        self._previous = previous
        self.__cypher = None
        self._index_node = index_node
        self._obj = obj
        if previous is not None:
            if self._index_node is None:
                if isinstance(previous, ObjectQuery):
                    self._index_node = previous._node
                elif self._index_node != 'start':
                    self._index_node = previous._index_node
            if obj is None:
                self._obj = previous._obj
        if start is None:
            self._start = self
        else:
            self._start = start
        if self._obj is not None:
            self._obj = self._normalise_object(self._obj)[0]
        self._names = [] if names is None else names

    def _get_object_of(self, maybe_attribute: str) -> Tuple[str, bool, bool]:
        """
        Given a str that might refer to an attribute, return the object that contains that attribute
        and also whether the attribute and the object are singular
        inferred obj | attr
            s        |  s      (i.e. run.obid -> run.ob.obid)
            p        |  p      (i.e. ...-> run.l1specs.snrs) - cannot be inferred just from a name
            s        |  p      (i.e. run.obids -> run.ob.obids)
            p        |  s      (i.e. run.snr -> run.l1spec.snr) - is not allowed
         :return: obj, obj_is_singular, attr_is_singular
        """
        single_name = self._data.singular_name(maybe_attribute)
        hs = {h.__name__ for h in self._data.factor_hierarchies[single_name]}
        if len(hs) > 1:
            raise AmbiguousPathError(f"There are multiple attributes called {maybe_attribute} with the following parent objects: {hs}."
                                     f" Please be specific e.g. `{hs.pop()}.{maybe_attribute}`")
        obj = hs.pop()
        if self._obj is None:
            return obj, True, False
        if self._obj == obj:
            return self._obj, True, self._data.is_singular_name(maybe_attribute)
        if not self._data.is_factor_name(maybe_attribute):
            raise ValueError(f"{maybe_attribute} is not a valid attribute name")
        path, obj_is_singular = self._data.path_to_hierarchy(self._obj, obj, False)
        attr_is_singular = self._data.is_singular_name(maybe_attribute)
        if obj_is_singular and attr_is_singular:
            pass  # run.obid -> run.ob.obid
        elif not obj_is_singular and not attr_is_singular:
            pass
        elif obj_is_singular and not attr_is_singular:
            pass  # run.obids -> run.ob.obids
        else:
            # ob.runid -> Error
            plural_name = self._data.plural_name(obj)
            original = self._data.singular_name(self._obj)
            raise CardinalityError(f"Requested one `{single_name}` from `{original}` "
                                   f"when `{original}` has several `{plural_name}`")
        return obj, obj_is_singular, attr_is_singular

    def _normalise_object(self, obj: str):
        obj = obj.lower()
        try:
            h = self._data.singular_hierarchies[obj]
            singular = True
        except KeyError:
            h = self._data.plural_hierarchies[obj]
            singular = False
        return h.__name__, singular

    @property
    def _cypher(self):
        if self.__cypher is None:
            self.__cypher = self._G.cypher_lines(self._node)
        return self.__cypher

    @property
    def _result_graph(self):
        return self._G.restricted(self._node)

    def _plot_graph(self, fname):
        return self._G.export(fname, self._node)

    @classmethod
    def _spawn(cls, parent: 'BaseQuery', node, obj=None, index_node=None, single=False, *args, **kwargs):
        return cls(parent._data, parent._G, node, parent, obj, parent._start, index_node, single, *args, **kwargs)

    def _get_path_to_object(self, obj, want_single) -> Tuple[str, bool]:
        return self._data.path_to_hierarchy(self._obj, obj, want_single)

    def _slice(self, slc):
        """
        obj[slice]
        filter, shrinks, destructive
        filter by using HEAD/TAIL/SKIP/LIMIT
        e.g. obs.runs[:10] will return the first 10 runs for each ob (in whatever order they were in)
        you must use query(skip, limit) to request a specific number of rows in total since this is unrelated to the actual query
        """
        raise NotImplementedError

    def _filter_by_mask(self, mask):
        """
        obj[boolean_filter]
        filter, shrinks, destructive
        filters based on a list of True/False values constructed beforehand, parentage of the booleans must be derived from the obj
        e.g. `ob.l1stackedspectra[ob.l1stackedspectra.camera == 'red']` gives only the red stacks
             `ob.l1stackedspectra[ob.l1singlespectra == 'red']` is invalid since the lists will not be the same size or have the same parentage
        """
        n = self._G.add_filter(self._node, mask._node, direct=False)
        return self.__class__._spawn(self, n, single=self._single)

    def _aggregate(self, wrt, string_op, predicate=False):
        if wrt is None:
            wrt = self._start
        if predicate:
            n = self._G.add_predicate_aggregation(self._node, wrt._node, string_op)
        else:
            n = self._G.add_aggregation(self._node, wrt._node, string_op)
        from .objects import AttributeQuery
        return AttributeQuery._spawn(self, n, wrt._obj, wrt._node, single=True)
