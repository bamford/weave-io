from copy import copy, deepcopy

from weaveio.basequery.query import Node, Path, Generator, Branch, Predicate, FullQuery


# data.runs[runid].exposure.runs.vphs

from weaveio.basequery.query import NodeProperty

from weaveio.data import OurData

data = OurData('data', port=11007)
thing = data.exposures.runs.exposures.runs['runids', 'expmjd', 'cnames']
print(thing.query.to_neo4j()[0])
# print(thing())