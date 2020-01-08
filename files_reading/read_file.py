from . import ibw_files
from . import xrdml_files
from . import ardf_files
import h5py
import os

def tohdf5(filename):
    if type(filename) == list:
        merge_hdf5(filename, 'merged_file', erase_file='partial')
    else:
        if filename.split('.')[-1] == 'ibw':
            ibw_files.ibw2hdf5(filename)
        elif filename.split('.')[-1] == 'xrdml':
            xrdml_files.xrdml2hdf5(filename)
        elif filename.split('.')[-1] == 'ardf' or filename.split('.')[-1] == 'ARDF':
            ardf_files.ardf2hdf5(filename)
        else:
            print('file type not yet supported')

def merge_hdf5(filelist, combined_name, erase_file='partial'):
    i = 0
    temporary = False
    if erase_file == 'all':
        print('CAUTION YOU RISK ERASING SOME RAW DATA FILES, ARE YOU SURE ? Due to the way python erase file, they will NOT be in the recycle bin. Please write "Yes, I am sure!" if you want to take that risk')
        tmp_val = input()
        if tmp_val != 'Yes, I am sure!':
              return 
              
    with h5py.File(combined_name +'.hdf5', 'w') as f:
        typegrp = f.create_group('type')
        metadatagrp = f.create_group('metadata')
        datagrp = f.create_group('datasets')
        procgrp = f.create_group('process')   
        #print(f.keys())

    for filename in filelist:
        if filename.split('.')[0] == combined_name:
            print('Warning: \''+filename+'\' already exists. File has been ignored and will be overwritten.')
            continue
        else:
            if filename.split('.')[-1] != 'hdf5': 
                if not (filename.split('.')[0] + '.hdf5' in filelist):
                    try:
                        tohdf5(filename)
                        if erase_file == 'all':
                            os.remove(filename)
                        filename = filename.split('.')[0] + '.hdf5'
                        temporary = True
                    except:
                        print('Warning: \''+filename+'\' is a non-hdf5 file. File has been ignored.')
                        continue
                else:
                    print('Warning: \''+filename+'\' is a non-hdf5 file, but the list containt a hdf5 file with the same name. File has been ignored.')
                    continue
        with h5py.File(combined_name +'.hdf5', 'a') as f:
            #print(f.keys())
            typegrp = f['type']
            metadatagrp = f['metadata']
            datagrp = f['datasets']
            procgrp = f['process']   
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
        #print(erase_file)    
        if erase_file == 'all':
            print('Erasing the file')
            os.remove(filename)
        elif erase_file == 'partial' and temporary:
            temporary = False
            os.remove(filename)
            
                
    print(str(i)+'/'+str(len(filelist))+' files merged into \'' + combined_name +'.hdf5\'')
