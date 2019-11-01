from . import ibw_files
from . import xrdml_files
from . import ardf_files

def tohdf5(filename):
    if filename.split('.')[-1] == 'ibw':
        ibw_files.ibw2hdf5(filename)
    elif filename.split('.')[-1] == 'xrdml':
        xrdml_files.xrdml2hdf5(filename)
    elif filename.split('.')[-1] == 'ardf':
        ardf_files.ardf2hdf5(filename)
    else:
        print('file type not yet supported')


