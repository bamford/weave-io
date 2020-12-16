from dataclasses import dataclass
from string import ascii_letters, ascii_uppercase, digits, printable
from typing import Dict, List, Any

import hypothesis.strategies as st
from hypothesis import given, note, settings, reproduce_failure
from hypothesis.strategies import builds, data, composite
from pytest import mark
import numpy as np

@composite
def _key_strategy(draw, start=st.text(min_size=1, max_size=1, alphabet=ascii_letters),
                 tail=st.text(min_size=0, alphabet=ascii_letters+digits)):
    return draw(start) + draw(tail)
key_strategy = _key_strategy()


@composite
def _label_strategy(draw, start=st.text(min_size=1, max_size=1, alphabet=ascii_uppercase),
                 tail=st.text(min_size=1, alphabet=ascii_letters+digits)):
    return draw(start) + draw(tail)
label_strategy = _label_strategy()

labels_strategy = st.lists(label_strategy, min_size=1, max_size=5, unique=True)

_basic_types = [st.integers(min_value=-9999, max_value=999), st.text(alphabet=ascii_letters+digits), st.floats(allow_nan=False, allow_infinity=False)]
_list_types = [st.lists(x) for x in _basic_types]

baseproperty_strategy = st.one_of(*_basic_types)
neo4jproperty_strategy = st.one_of(*_basic_types+_list_types)
properties_strategy = st.dictionaries(key_strategy, neo4jproperty_strategy)
identproperties_strategy = st.dictionaries(key_strategy, baseproperty_strategy)


@dataclass
class Node:
    labels: List[str]
    properties: Dict[str, Any]
    identproperties: Dict[str, Any]

    @property
    def allproperties(self):
        props = self.properties.copy()
        props.update(self.identproperties)
        return props


@dataclass
class Relation:
    reltype: str
    properties: Dict[str, Any]
    identproperties: Dict[str, Any]


node_strategy = builds(Node, labels=labels_strategy, properties=properties_strategy, identproperties=identproperties_strategy)
rel_strategy = builds(Relation, reltype=label_strategy, properties=properties_strategy, identproperties=identproperties_strategy)


def labels_from_labels(labels: List, how, sampler):
    if how == False:
        return labels
    elif how == 'entire':  # must be entirely different
        return sampler.draw(labels_strategy.filter(lambda xs: not any(x in labels for x in xs)))
    elif how == 'crop':  # return the same list up until an index
        if len(labels) == 1:
            return labels  # odd, but we need to ignore it here
        index = sampler.draw(st.integers(min_value=1, max_value=len(labels) - 1))
        return labels[:index]
    elif how == 'extend':  # just add more labels
        return labels + sampler.draw(labels_strategy.filter(lambda xs: not any(x in labels for x in xs)
                                                                       and len(xs) > 1))
    elif how == 'crop&extend':
        if len(labels) == 1:
            return labels
        return labels[:-1] + [sampler.draw(label_strategy.filter(lambda x: x not in labels[:-1]))]
    else:
        raise ValueError(f"Method to change labels {how} is unknown")


def properties_from_properties(properties: Dict, how, sampler, properties_strategy_type, property_strategy_type, exclude_keys=None):
    if exclude_keys is None:
        exclude_keys = []
    exclude_keys = exclude_keys.copy()
    new = properties.copy()
    keys = list(properties.keys())
    if how == False:
        return properties
    elif how == 'entirekeys':  # must be entirely different
        exclude_keys += list(properties.keys())
        return sampler.draw(properties_strategy_type.filter(lambda d: not any(k in exclude_keys for k in d)))
    elif how == 'crop':  # return the same list up until an index
        if len(properties) == 0:
            return {}  # this situation is handled by 'extend'
        indexes = sampler.draw(st.lists(st.integers(min_value=0, max_value=len(keys) - 1), unique=True, min_size=1, max_size=len(keys)))
        for i in indexes:
            del new[keys[i]]
        return new
    elif how == 'addkeys':  # just add more labels
        exclude_keys += list(properties.keys())
        strat = properties_strategy_type.filter(lambda x: not any(k in exclude_keys for k in x) and len(x))
        additions = sampler.draw(strat)
        new.update(additions)
        return new
    elif how == 'overwritekeys':
        if len(properties) == 0:
            return {}  # this situation is handled by 'extend'
        n = sampler.draw(st.integers(min_value=1, max_value=len(properties)))
        whichkeys = sampler.draw(st.lists(st.integers(min_value=0, max_value=len(keys)-1), min_size=n, max_size=n, unique=True))
        newkeys = [keys[i] for i in whichkeys]
        oldvalues = [properties[k] for k in newkeys]
        newvalues = sampler.draw(st.lists(property_strategy_type.filter(lambda x: x not in oldvalues),
                                          min_size=n, max_size=n)
                                 )
        for k, v in zip(newkeys, newvalues):
            new[k] = v
        return new
    else:
        raise ValueError(f"Method to change properties {how} is unknown")


def create_node_from_node(node, sampler, different_labels, different_properties, different_identproperties):
    new_identproperties = properties_from_properties(node.identproperties, different_identproperties,
                                                     sampler, identproperties_strategy, baseproperty_strategy)
    new_properties = properties_from_properties(node.properties, different_properties, sampler,
                                                properties_strategy, neo4jproperty_strategy,
                                                list(new_identproperties.keys()))
    new_labels = labels_from_labels(node.labels, different_labels, sampler)
    return Node(new_labels, identproperties=new_identproperties, properties=new_properties)


@given(labels=labels_strategy.filter(lambda x: len(x) > 1), sampler=data())
@mark.parametrize('different_labels', [False, 'entire', 'crop', 'extend', 'crop&extend'])
def test_labels_from_labels(labels, different_labels, sampler):
    newlabels = labels_from_labels(labels, different_labels, sampler)
    note(f"newlabels = {newlabels}")
    note(f"oldlabels = {labels}")
    assert len(newlabels) > 0
    assert len(set(newlabels)) == len(newlabels)
    if different_labels == False:
        assert newlabels == labels
    elif different_labels == 'entire':
        assert not any(l in labels for l in newlabels)
    elif different_labels == 'crop':
        assert newlabels == labels[:len(newlabels)]
    elif different_labels == 'extend':
        assert newlabels[:len(labels)] == labels
        assert len(newlabels) > len(labels)
    elif different_labels == 'crop&extend':
        assert newlabels[0] == labels[0]  # at least one is the same


def is_null(x):
    try:
        return np.all(x != x)
    except TypeError:
        pass
    return x is None


def is_equal_or_null(x, y):
    return x == y or (is_null(x) and is_null(y))


@settings(print_blob=True)
@given(properties=properties_strategy, sampler=data())
@mark.parametrize('different_properties', [False, 'entirekeys', 'crop', 'addkeys', 'overwritekeys'])
def test_properties_from_properties(properties, different_properties, sampler):
    excluded = ['a', 'b']
    newprops = properties_from_properties(properties, different_properties, sampler,
                                          properties_strategy, neo4jproperty_strategy, excluded)
    note(f"newprops = {newprops}")
    note(f"oldprops = {properties}")
    if different_properties == False:
        assert properties == newprops
    elif different_properties == 'entirekeys':
        assert not any(k in properties for k in newprops)
        assert not any(e in newprops for e in excluded)
    elif different_properties == 'crop':
        assert len(newprops) < len(properties) or len(properties) == 0
        assert all(is_equal_or_null(properties[k], v) for k, v in newprops.items())
    elif different_properties == 'addkeys':
        assert all(is_equal_or_null(newprops[k], v) for k, v in properties.items())
        assert len(newprops) > len(properties)
        assert not any(e in newprops for e in excluded)
    elif different_properties == 'overwritekeys':
        assert len(newprops) == len(properties)
        if len(properties) > 0:
            # at least one different
            assert any(not is_equal_or_null(newprops[k], v) for k, v in properties.items())

