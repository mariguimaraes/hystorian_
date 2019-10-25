import ibw_files
import xrdml_files
import ardf_files

def tohdf5(filename):
    if filename.split('.')[-1] == 'ibw':
        ibw2hdf5(filename)
    elif filename.split('.')[-1] == 'xrdml':
        xrdml2hdf5(filename)
    elif filename.split('.')[-1] == 'ardf':
        ardf2hdf5(filename)
    else:
        print('file type not yet supported')


