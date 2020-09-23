import inspect
from pathlib import Path
from typing import Union

import networkx as nx
import xxhash
from astropy.io import fits
from astropy.table import Table
from graphviz import Source
from tqdm import tqdm
import pandas as pd

from .config_tables import progtemp_config
from .graph import Graph, Node, Relationship, ContextError


def graph2pdf(graph, ftitle):
    dot = nx.nx_pydot.to_pydot(graph)
    dot.set_strict(False)
    # dot.obj_dict['attributes']['splines'] = 'ortho'
    dot.obj_dict['attributes']['nodesep'] = '0.5'
    dot.obj_dict['attributes']['ranksep'] = '0.75'
    dot.obj_dict['attributes']['overlap'] = False
    dot.obj_dict['attributes']['penwidth'] = 18
    dot.obj_dict['attributes']['concentrate'] = False
    Source(dot).render(ftitle, cleanup=True, format='pdf')


lightblue = '#69A3C3'
lightgreen = '#71C2BF'
red = '#D08D90'
orange = '#DFC6A1'
purple = '#a45fed'
pink = '#d50af5'

hierarchy_attrs = {'type': 'hierarchy', 'style': 'filled', 'fillcolor': red, 'shape': 'box', 'edgecolor': red}
abstract_hierarchy_attrs = {'type': 'hierarchy', 'style': 'filled', 'fillcolor': red, 'shape': 'box', 'edgecolor': red}
factor_attrs = {'type': 'factor', 'style': 'filled', 'fillcolor': orange, 'shape': 'box', 'edgecolor': orange}
identity_attrs = {'type': 'id', 'style': 'filled', 'fillcolor': purple, 'shape': 'box', 'edgecolor': purple}
product_attrs = {'type': 'factor', 'style': 'filled', 'fillcolor': pink, 'shape': 'box', 'edgecolor': pink}
l1file_attrs = {'type': 'file', 'style': 'filled', 'fillcolor': lightblue, 'shape': 'box', 'edgecolor': lightblue}
l2file_attrs = {'type': 'file', 'style': 'filled', 'fillcolor': lightgreen, 'shape': 'box', 'edgecolor': lightgreen}
rawfile_attrs = l1file_attrs


class Multiple:
    def __init__(self, node, minnumber=1, maxnumber=None):
        self.node = node
        self.minnumber = minnumber
        self.maxnumber = maxnumber
        self.name = node.plural_name
        self.singular_name = node.singular_name
        self.plural_name = node.plural_name
        try:
            self.factors =  self.node.factors
        except AttributeError:
            self.factors = []
        try:
            self.parents = self.node.parents
        except AttributeError:
            self.parents = []

    def __repr__(self):
        return f"<Multiple({self.node} [{self.minnumber} - {self.maxnumber}])>"


class PluralityMeta(type):
    def __new__(meta, name, bases, dct):
        if dct.get('plural_name', None) is None:
            dct['plural_name'] = name.lower() + 's'
        dct['singular_name'] = name.lower()
        dct['plural_name'] = dct['plural_name'].lower()
        dct['singular_name'] = dct['singular_name'].lower()
        r = super(PluralityMeta, meta).__new__(meta, name, bases, dct)
        return r


class Graphable(metaclass=PluralityMeta):
    idname = None
    name = None
    identifier = None
    indexers = []
    type_graph_attrs = {}
    plural_name = None
    singular_name = None

    @property
    def neotypes(self):
        clses = [i.__name__ for i in inspect.getmro(self.__class__)]
        clses = clses[:clses.index('Graphable')]
        return clses

    @property
    def neoproperties(self):
        if self.idname is not None:
            d = {self.idname: self.identifier}
            d['id'] = self.identifier
            return d
        else:
            return {'dummy': 1}   # just to stop py2neo complaining, shouldnt actually be encountered

    def __init__(self, **nodes):
        try:
            tx = Graph.get_context().tx
            self.node = Node(*self.neotypes, **self.neoproperties)
            try:
                key = list(self.neoproperties.keys())[0]
            except IndexError:
                key = None
            primary = {'primary_label': self.neotypes[-1], 'primary_key': key}
            tx.merge(self.node, **primary)
            for k, node_list in nodes.items():
                for node in node_list:
                    if k in [i.lower() for i in self.indexers]:
                        type = 'indexes'
                    else:
                        type = 'is_required_by'
                    tx.merge(Relationship(node.node, type, self.node))
        except ContextError:
            pass


class Factor(Graphable):
    type_graph_attrs = factor_attrs

    @property
    def neotypes(self):
        return ['Factor', self.idname]

    @property
    def neoproperties(self):
        return {'value': self.identifier, 'id': self.identifier}

    def __init__(self, name, value, plural_name=None):
        self.idname = name
        self.identifier = value
        self.name = f"{self.idname}({self.identifier})"
        super(Factor, self).__init__()

    def __repr__(self):
        return f"<Factor({self.idname}={self.identifier})>"


class Hierarchy(Graphable):
    idname = None
    parents = []
    factors = []
    indexers = []
    type_graph_attrs = hierarchy_attrs

    def __getattr__(self, item):
        for p in self._kwargs.values():
            try:
                return getattr(p, item)
            except AttributeError:
                continue
        raise AttributeError(f"{item} cannot be found in {self} or its parent structure")


    def __repr__(self):
        return self.name

    def __init__(self, **kwargs):
        if self.idname not in kwargs and self.idname is not None:
            kwargs[self.idname] = ''.join(str(kwargs[f.lower()]) for f in self.factors)
        parents = {p.__name__.lower() if isinstance(p, type) else p.name: p for p in self.parents}
        factors = {f.lower(): f for f in self.factors}
        specification = parents.copy()
        specification.update(factors)
        if self.idname is not None:
            self.identifier = kwargs.pop(self.idname)
            setattr(self, self.idname, self.identifier)
        self._kwargs = kwargs.copy()

        predecessors = {}
        for name, nodetype in specification.items():
            value = kwargs.pop(name)
            setattr(self, name, value)
            if isinstance(nodetype, Multiple):
                if not isinstance(value, (tuple, list)):
                    raise TypeError(f"{name} expects multiple elements")
                if name in factors:
                    value = [Factor(name, val) for val in value]
            elif name in factors:
                value = [Factor(name, value)]
            else:
                value = [value]
            predecessors[name] = value
        if len(kwargs):
            raise KeyError(f"{kwargs.keys()} are not relevant to {self.__class__}")
        if self.idname is not None:
            self.name = f"{self.__class__.__name__}({self.idname}={self.identifier})"
        else:
            name = xxhash.xxh32()
            for predecessor_list in predecessors.values():
                for predecessor in predecessor_list:
                    name.update(predecessor.name)
            name = '#' + name.hexdigest()
            self.idname = 'id'
            self.identifier = name
            self.name = f"{self.__class__.__name__}({name})"
        super(Hierarchy, self).__init__(**predecessors)


class File(Graphable):
    idname = 'fname'
    constructed_from = []
    indexable_by = []
    type_graph_attrs = l1file_attrs

    def __init__(self, fname: Union[Path, str], **kwargs):
        self.fname = Path(fname)
        self.identifier = str(self.fname)
        self.name = f'{self.__class__.__name__}({self.fname})'
        if len(kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            self.predecessors = kwargs
        else:
            self.predecessors = self.read()
        super(File, self).__init__(**self.predecessors)
        self.index = None

    def match(self, directory: Path):
        raise NotImplementedError

    @property
    def graph_name(self):
        return f"File({self.fname})"

    def read(self):
        raise NotImplementedError

    def build_index(self) -> None:
        self.index['rowid'] = range(len(self.index))
        self.index['fname'] = self.fname

    def match_index(self, index) -> pd.DataFrame:
        self.build_index()
        keys = [i for i in index.columns if i not in ['fname', 'rowid']]
        filt = self.index[keys].isin(index[keys])
        return self.index[filt]



class ArmConfig(Hierarchy):
    factors = ['Resolution', 'VPH', 'Camera']
    idname = 'armcode'

    @classmethod
    def from_progtemp_code(cls, progtemp_code):
        config = progtemp_config.loc[progtemp_code[0]]
        red = cls(resolution=str(config.resolution), vph=int(config.red_vph), camera='red')
        blue = cls(resolution=str(config.resolution), vph=int(config.blue_vph), camera='blue')
        return red, blue


class ObsTemp(Hierarchy):
    factors = ['MaxSeeing', 'MinTrans', 'MinElev', 'MinMoon', 'MaxSky']
    idname = 'obstemp'

    @classmethod
    def from_header(cls, header):
        names = [f.lower() for f in cls.factors]
        return cls(**{n: v for v, n in zip(list(header['OBSTEMP']), names)})


class Target(Hierarchy):
    idname = 'cname'

    @classmethod
    def from_fibinfo_row(cls, row):
        return Target(cname=row['CNAME'])


class TargetSet(Hierarchy):
    parents = [Multiple(Target)]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_fibinfo(cls, fibinfo):
        targets = [Target.from_fibinfo_row(row) for row in tqdm(fibinfo[:10])]
        return cls(targets=targets)


class ProgTemp(Hierarchy):
    factors = ['Mode', 'Binning']
    parents = [Multiple(ArmConfig, 2, 2)]

    @classmethod
    def from_progtemp_code(cls, progtemp_code):
        progtemp_code = progtemp_code.split('.')[0]
        progtemp_code = list(map(int, progtemp_code))
        configs = ArmConfig.from_progtemp_code(progtemp_code)
        mode = progtemp_config.loc[progtemp_code[0]]['mode']
        binning = progtemp_code[3]
        return cls(mode=mode, binning=binning, armconfigs=configs)


class OBSpec(Hierarchy):
    factors = ['OBTitle']
    parents = [ObsTemp, TargetSet, ProgTemp]


class OBRealisation(Hierarchy):
    idname = 'obid'
    factors = ['OBStartMJD']
    parents = [OBSpec]


class Exposure(Hierarchy):
    parents = [OBRealisation]
    factors = ['ExpMJD']


class Run(Hierarchy):
    idname = 'runid'
    parents = [Exposure]
    factors = ['Camera']
    indexers = ['Camera']


class HeaderFibinfoFile(File):
    fibinfo_i = -1

    def read(self):
        header = fits.open(self.fname)[0].header
        runid = str(header['RUN'])
        camera = str(header['CAMERA'].lower()[len('WEAVE'):])
        expmjd = str(header['MJD-OBS'])
        res = str(header['VPH']).rstrip('123')
        obstart = str(header['OBSTART'])
        obtitle = str(header['OBTITLE'])
        obid = str(header['OBID'])

        fibinfo = Table(fits.open(self.fname)[self.fibinfo_i].data)
        progtemp = ProgTemp.from_progtemp_code(header['PROGTEMP'])
        vph = int(progtemp_config[(progtemp_config['mode'] == progtemp.mode)
                              & (progtemp_config['resolution'] == res)][f'{camera}_vph'].iloc[0])
        armconfig = ArmConfig(vph=vph, resolution=res, camera=camera)  # must instantiate even if not used
        obstemp = ObsTemp.from_header(header)
        targetset = TargetSet.from_fibinfo(fibinfo)
        obspec = OBSpec(targetset=targetset, obtitle=obtitle, obstemp=obstemp, progtemp=progtemp)
        obrealisation = OBRealisation(obid=obid, obstartmjd=obstart, obspec=obspec)
        exposure = Exposure(expmjd=expmjd, obrealisation=obrealisation)
        run = Run(runid=runid, camera=camera, exposure=exposure)
        return {'run': [run]}

    def build_index(self) -> None:
        if self.index is None:
            self.index = pd.DataFrame({'cname': [i.cname for i in self.targets]})
        super(HeaderFibinfoFile, self).build_index()


class Raw(HeaderFibinfoFile):
    parents = [Run]
    fibinfo_i = 3

    @classmethod
    def match(cls, directory: Path):
        return directory.glob('r*.fit')


class L1Single(HeaderFibinfoFile):
    parents = [Run]
    constructed_from = [Raw]

    @classmethod
    def match(cls, directory):
        return directory.glob('single_*.fit')


class L1Stack(HeaderFibinfoFile):
    parents = [OBRealisation]
    factors = ['VPH']
    constructed_from = [L1Single]

    @classmethod
    def match(cls, directory):
        return directory.glob('stacked_*.fit')


class L1SuperStack(File):
    parents = [OBSpec]
    factors = ['VPH']
    constructed_from = [L1Single]

    @classmethod
    def match(cls, directory):
        return directory.glob('superstacked_*.fit')


class L1SuperTarget(File):
    parents = [ArmConfig, Target]
    factors = ['Binning', 'Mode']
    constructed_from = [L1Single]

    @classmethod
    def match(cls, directory):
        return directory.glob('[Lm]?WVE_*.fit')


class L2Single(File):
    parents = [Exposure]
    constructed_from = [Multiple(L1Single, 2, 2)]

    @classmethod
    def match(cls, directory):
        return directory.glob('single_*_aps.fit')


class L2Stack(File):
    parents = [Multiple(ArmConfig, 1, 3), TargetSet]
    factors = ['Binning', 'Mode']
    constructed_from = [Multiple(L1Stack, 0, 3), Multiple(L1SuperStack, 0, 3)]

    @classmethod
    def match(cls, directory):
        return directory.glob('(super)?stacked_*_aps.fit')


class L2SuperTarget(File):
    parents = [Multiple(ArmConfig, 1, 3), Target]
    factors = ['Mode', 'Binning']
    constructed_from = [Multiple(L1SuperTarget, 2, 3)]

    @classmethod
    def match(cls, directory):
        return directory.glob('[Lm]?WVE_*_aps.fit')
