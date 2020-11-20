import h5py
import importlib
from inspect import getmembers, isfunction
from multiscale import io, processing
import inspect
import numpy as np

def my_import(name):
    components = name.split('.')
    print(components)
    mod = __import__(components[0])
    print(mod)
    for comp in components[1:]:
        mod = getattr(mod, comp)
        print(mod)
    return mod

def compress_hdf5(file, error_threshold=0, bypass_verification=False):
    if not bypass_verification:
        print('Please understand that this function WILL remove processed datas from your hdf5.\n '
              'Normally they should be rebuildable using existing functions. But no guarantee is given. \n '
              'Please write "I understand the risk"')
        input_value = input()
        if input_value != 'I understand the risk':
            return -1

    #print([o for o in getmembers(multiscale.io) if isfunction(o[1])])
    with h5py.File(file, 'r+') as f:
        processes = f['process']
        processing_lst = processes.keys()

        for k in list(processing_lst)[::-1]:
            module = importlib.import_module('.'.join(processes[k]['SD_P4_zB5_050mV_-2550mV_0002/Phase2Retrace'].attrs['operation name'].split('.')[:-1]))
            func = getattr(module, processes[k]['SD_P4_zB5_050mV_-2550mV_0002/Phase2Retrace'].attrs['operation name'].split('.')[-1])
            print(func.__name__)
            result = func(f[processes[k]['SD_P4_zB5_050mV_-2550mV_0002/Phase2Retrace'].attrs['source'][0]][()])
            print(processes[k]['SD_P4_zB5_050mV_-2550mV_0002/Phase2Retrace'][()])
            print(f[processes[k]['SD_P4_zB5_050mV_-2550mV_0002/Phase2Retrace'].attrs['source'][0]][()])
            print(result)
            if (result['data'] == processes[k]['SD_P4_zB5_050mV_-2550mV_0002/Phase2Retrace'][()]).all():
                print('OK')

                tmpattrs = {}
                for k2 in processes[k]['SD_P4_zB5_050mV_-2550mV_0002/Phase2Retrace'].attrs.keys():
                    (processes[k]['SD_P4_zB5_050mV_-2550mV_0002/Phase2Retrace'].attrs[k2])
                    tmpattrs[k2] = processes[k]['SD_P4_zB5_050mV_-2550mV_0002/Phase2Retrace'].attrs[k2]

                del(processes[k]['SD_P4_zB5_050mV_-2550mV_0002/Phase2Retrace'])
                processes[k]['SD_P4_zB5_050mV_-2550mV_0002'].create_dataset('Phase2Retrace', (1,), dtype=int)
                for k3 in tmpattrs.keys():
                    val = tmpattrs[k3]
                    processes[k]['SD_P4_zB5_050mV_-2550mV_0002/Phase2Retrace'].attrs.__setitem__(k3,val)
            else:
                print('NOT OK')

    processing.core.deallocate_hdf5_memory(file, verify=False)



