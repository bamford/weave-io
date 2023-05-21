from pathlib import Path
from tqdm import tqdm
from functools import partialmethod

from weaveio.file import File
from weaveio.opr3 import Data

import logging
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def match_files(filetype, directory, night):
    """Returns all matching files within a directory"""
    return (f for f in Path(directory).rglob(f'*{night}/*.fit*') if filetype.match_file(directory, f, None))


def find_files(data, night):
    filelist = []
    for filetype in data.filetypes:
        filelist += sorted([i for i in match_files(filetype, data.rootdir, night)], key=lambda f: f.name)
    return [f for f in filelist if File.check_mos(f)]


def main(night):
    data = Data()
    logger.info(f'Preparing to ingest data for night {night}')
    with data.write:
        fs = find_files(data, night)
        if len(fs) == 0:
            logger.info(f'No MOS files found')
        else:
            data.write_files(*fs, timeout=60*60, debug=False, test_one=False, debug_time=False,
                             debug_params=False, dryrun=False)


if __name__ == '__main__':
    import sys
    main(sys.argv[1])
