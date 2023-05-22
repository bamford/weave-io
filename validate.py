from tqdm import tqdm
from functools import partialmethod

from weaveio.opr3 import Data

import logging
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


if __name__ == '__main__':
    data = Data()
    logger.info(f'Preparing to validate data')
    data.validate()
