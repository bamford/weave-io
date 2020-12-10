from weaveio.data import Data
from weaveio.opr3.file import RawFile, L1SingleFile, L1StackFile, L1SuperStackFile, L1SuperTargetFile, L2File


class OurData(Data):
    filetypes = [RawFile, L1SingleFile, L1StackFile, L1SuperStackFile, L1SuperTargetFile, L2File]