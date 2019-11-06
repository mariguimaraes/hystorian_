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
                    for source_type in source_f['type']:
                        source_f.copy('type/'+source_type, typegrp)
                    for source_metadata in source_f['metadata']:
                        source_f.copy('metadata/'+source_metadata, metadatagrp)
                    for source_data in source_f['datasets']:
                        source_f.copy('datasets/'+source_data, datagrp)
                    try:
                        for source_proc in source_f['process']:
                            source_f.copy('process/'+source_proc, procgrp)
                    except:
                        pass
                    i = i + 1
                print('\''+filename+'\' successfully merged')
    print(str(i)+'/'+str(len(filelist))+' files merged into \'' + combined_name +'.hdf5\'')
