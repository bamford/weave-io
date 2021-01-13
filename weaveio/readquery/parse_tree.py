from collections import defaultdict
from functools import reduce
from operator import and_
from typing import List

import networkx as nx

from weaveio.readquery.tree import plot, Alignment, BranchHandler, TraversalPath


def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])


def print_nested_list(nested, tab=0):
    for entry in nested:
        if isinstance(entry, list):
            print_nested_list(['--'] + entry, tab+1)
        else:
            print('    '*tab, entry)


def shared_hierarchy_branch(graph, branch):
    branches = reduce(and_, [set(p.find_hierarchy_branches()) for p in branch.parents])
    distances = [(b, nx.shortest_path_length(graph, b, branch)) for b in branches]
    return max(distances, key=lambda x: x[1])[0]


def parse(graph) -> List:
    aligns = [i for i in graph.nodes if isinstance(i.action, Alignment)][::-1]
    shared_aligns = defaultdict(list)
    for align in aligns:
        shared_aligns[shared_hierarchy_branch(graph, align)].append(align)
    query = []
    todo = list(nx.algorithms.topological_sort(graph))
    while todo:
        node = todo.pop(0)
        if node in shared_aligns:
            align_list = shared_aligns[node][:1] # first one only??
            for align in align_list:
                inputs = align.action.branches
                inputs += (align.action.reference, )
                subqueries = []
                for input_node in inputs:
                    before = list(nx.descendants(graph, node)) + [node]
                    after = list(nx.ancestors(graph, input_node)) + [input_node]
                    newgraph = nx.subgraph_view(graph, lambda n: n in before and n in after)
                    subquery = parse(newgraph)
                    subqueries.append(subquery)
                done = list(set(flatten(subqueries)))
                for d in done:
                    if d in todo:
                        del todo[todo.index(d)]
                reference_subquery = subqueries.pop(-1)
                query += subqueries
                query += reference_subquery
        else:
            query.append(node)
    return query


if __name__ == '__main__':
    handler = BranchHandler()
    ob = handler.begin('OB')
    target = ob.traverse(TraversalPath('->', 'target'))
    run = ob.traverse(TraversalPath('->', 'run'))
    exposure = ob.traverse(TraversalPath('->', 'exposure'))
    ob_targets = ob.collect([], [target])
    any_ob_targets = ob_targets.operate('{any}')
    ob_runs = ob.collect([], [run])
    any_ob_runs = ob_runs.operate('{any}')
    ob_exposures = ob.collect([], [exposure])
    any_ob_exposures = ob_exposures.operate('{any}')

    align0 = any_ob_runs.align(any_ob_targets)
    or1 = align0.operate('{or}')
    align1 = or1.align(any_ob_exposures)
    or2 = align1.operate('{or}')
    final = or2.filter('')
    final = final.results({final: [final.hierarchies[-1].get('obid')]})
    graph = final.relevant_graph
    plot(graph, '/opt/project/weaveio_example_querytree_test_branch.png')
    subqueries = parse(graph)
    print_nested_list(subqueries)

    print('================================================')

    from tree_test_weaveio_example import red_spectra
    final = red_spectra.results({})
    graph = final.relevant_graph
    plot(graph, '/opt/project/weaveio_example_querytree_red_branch.png')
    plot(final.accessible_graph, '/opt/project/weaveio_example_querytree_red_branch_accessible.png')
    subqueries = parse(graph)
    print_nested_list(subqueries)

