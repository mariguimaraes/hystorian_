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

def merge_hdf5(filelist, combined_name):
    i = 0
    with h5py.File(combined_name +'.hdf5', 'w') as f:
        typegrp = f.create_group('type')
        metadatagrp = f.create_group('metadata')
        datagrp = f.create_group('datasets')
        procgrp = f.create_group('process')
        
        for filename in filelist:
            if filename.split('.')[-1] != 'hdf5':
                print('Warning: \''+filename+'\' is a non-hdf5 file. File has been ignored.')
            elif filename.split('.')[0] == combined_name:
                print('Warning: \''+filename+'\' already exists. File has been ignored and will be overwritten.')
            else:
                with h5py.File(filename, 'r') as source_f:
                    source_f.copy('type/'+filename.split('.')[0], typegrp)
                    source_f.copy('metadata/'+filename.split('.')[0], metadatagrp)
                    source_f.copy('datasets/'+filename.split('.')[0], datagrp)
                    try:
                        source_f.copy('process/'+filename.split('.')[0], procgrp)
                    except:
                        pass
                    i = i + 1
                print('\''+filename+'\' successfully merged')
    print(str(i)+'/'+str(len(filelist))+' files merged into \'' + combined_name +'.hdf5\'')
