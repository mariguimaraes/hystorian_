try:
    xrdml_bool = True
    from . import xrdml_files

except:
    xrdml_bool = False

from . import ibw_files
from . import ardf_files
from . import sxm_files
from . import gsf_files

import h5py
import os


def tohdf5(filename):
    if type(filename) == list:
        merge_hdf5(filename, 'merged_file', erase_file='partial')
    else:
        if filename.split('.')[-1] == 'ibw':
            ibw_files.ibw2hdf5(filename)
        elif filename.split('.')[-1] == 'xrdml':
            if xrdml_bool:
                xrdml_files.xrdml2hdf5(filename)
            else:
                print('xrdml_files was not imported, probably due to the missing xrd_tools package. Please install it.')
        elif filename.split('.')[-1] == 'ardf' or filename.split('.')[-1] == 'ARDF':
            ardf_files.ardf2hdf5(filename)
        elif filename.split('.')[-1] == 'sxm':
            sxm_files.sxm2hdf5(filename)
        elif filename.split('.')[-1] == 'gsf':
            gsf_files.gsf2hdf5(filename)
        else:
            print('file type not yet supported')


def merge_hdf5(filelist, combined_name, erase_file='partial'):
    i = 0
    temporary = False

    with h5py.File(combined_name + '.hdf5', 'w') as f:
        typegrp = f.create_group('type')
        metadatagrp = f.create_group('metadata')
        datagrp = f.create_group('datasets')
        procgrp = f.create_group('process')
        # print(f.keys())

    for filename in filelist:
        if filename.split('.')[0] == combined_name:
            print('Warning: \'' + filename + '\' already exists. File has been ignored and will be overwritten.')
            continue
        else:
            if filename.split('.')[-1] != 'hdf5':
                if not (filename.split('.')[0] + '.hdf5' in filelist):
                    try:
                        tohdf5(filename)
                        filename = filename.split('.')[0] + '.hdf5'
                        temporary = True
                    except:
                        print('Warning: \'' + filename + '\' as impossible to convert. File has been ignored.')
                        continue
                else:
                    print(
                        'Warning: \'' + filename + '\' is a non-hdf5 file, but the list containt a hdf5 file with the same name. File has been ignored.')
                    continue
        with h5py.File(combined_name + '.hdf5', 'a') as f:
            # print(f.keys())
            typegrp = f['type']
            metadatagrp = f['metadata']
            datagrp = f['datasets']
            procgrp = f['process']
            with h5py.File(filename, 'r') as source_f:
                for source_type in source_f['type']:
                    source_f.copy('type/' + source_type, typegrp)
                for source_metadata in source_f['metadata']:
                    source_f.copy('metadata/' + source_metadata, metadatagrp)
                for source_data in source_f['datasets']:
                    source_f.copy('datasets/' + source_data, datagrp)
                try:
                    for source_proc in source_f['process']:
                        source_f.copy('process/' + source_proc, procgrp)
                except:
                    pass
                i = i + 1

        print('\'' + filename + '\' successfully merged')
        if erase_file == 'partial' and temporary:
            temporary = False
            if filename.split('.')[-1] == '.hdf5':
                os.remove(filename)

    print(str(i) + '/' + str(len(filelist)) + ' files merged into \'' + combined_name + '.hdf5\'')
