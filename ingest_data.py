from weaveio import *
from weaveio.opr3 import Data

logging.basicConfig(level=logging.INFO)

data = Data(dbname='bambase')
with data.write:
    #fs = data.find_files('raw', 'l1single', 'l1stack', 'l2single', 'l2stack')
    # fs is a list of filepaths
    for filetype in ('raw', 'L1', 'L2'):
        fs = list(data.rootdir.rglob(f'{filetype}/*/*100222[56789]*'))
        fs += list(data.rootdir.rglob(f'{filetype}/*/*1002230*'))
        parts = ['RR', 'GAND'] if filetype == 'L2' else None
        data.write_files(*fs, parts=parts,
                         timeout=10*60, debug=False, debug_time=True, dryrun=False)
data.validate()