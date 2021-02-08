import h5py
import importlib
from multiscale import processing
import ast
import sys
import numpy as np

def compress_hdf5(file, error_threshold=0, bypass_verification=False):
    identical = True
    if not bypass_verification:
        print('Please understand that this function WILL remove processed datas from your hdf5.\n '
              'Normally they should be rebuildable using existing functions. But no guarantee is given. \n '
              'Please write "I understand the risk"')
        input_value = input()
        if input_value != 'I understand the risk':
            return None

    with h5py.File(file, 'r+') as f:
        processes = f['process']
        processing_lst = processes.keys()
        for k in list(processing_lst)[::-1]:
            fname = list(processes[k].keys())[0]
            outputs = list(processes[k][fname])
            module = importlib.import_module('.'.join(
                processes[k][fname][outputs[0]].attrs['operation name'].split('.')[:-1]))

            func = getattr(module,
                           processes[k][fname][outputs[0]].attrs['operation name'].split('.')[-1])
            kwargs = {}
            if processes[k][fname].attrs.get('compressed') is None:
                for key in processes[k][fname][outputs[0]].attrs.keys():
                    if k.split('_')[0] == 'kwargs':
                        short_key = '_'.join(key.split('_')[1:])
                        kwargs[short_key] = processes[k][fname][outputs[0]].attrs[k]

                inputs = []
                for source in processes[k][fname][outputs[0]].attrs['source']:
                    inputs.append(f[source][()])
                return inputs
                input()
                #try:
                results = func(*inputs, **kwargs)
                if type(results) != tuple:
                    results = tuple([results])
                for i in range(len(outputs)):
                    if type(results[i]) == dict:
                        if 'hdf5_dict' in results[i]:
                            result_data = results[i]['data']
                        else:
                            result_data = results[i]
                    else:
                        result_data = results[i]
                    if (result_data - processes[k][fname][outputs[i]][()] < error_threshold).all():
                        print(func.__name__ + ': Result is not identical, not compressing')
                        identical = False
                        break
                #except Exception as e:
                #    print('ERROR: ' + e.__doc__)
                #    identical = False

                if identical:
                    for i in range(len(outputs)):
                        tmpattrs = {}
                        for k2 in processes[k][fname][outputs[i]].attrs.keys():
                            tmpattrs[k2] = processes[k][fname][outputs[i]].attrs[k2]

                        del (processes[k][fname][outputs[i]])
                        processes[k][fname].create_dataset(outputs[i], (1,), dtype=int)
                        for k3 in tmpattrs.keys():
                            val = tmpattrs[k3]
                            processes[k][fname][outputs[i]].attrs.__setitem__(k3, val)
                    processes[k][fname].attrs.__setitem__('compressed', True)
                else:
                    processes[k][fname].attrs.__setitem__('compressed', False)

    processing.core.deallocate_hdf5_memory(file, verify=False)

def decompress_hdf5(file):
    with h5py.File(file, 'r+') as f:
        processes = f['process']
        processing_lst = processes.keys()

        for k in list(processing_lst)[::-1]:
            fname = list(processes[k].keys())[0]
            if processes[k][fname].attrs.get('compressed'):
                outputs = list(processes[k][fname])
                module = importlib.import_module('.'.join(
                    processes[k][fname][outputs[0]].attrs['operation name'].split('.')[:-1]))

                func = getattr(module,
                               processes[k][fname][outputs[0]].attrs['operation name'].split('.')[-1])
                kwargs = {}
                for key in processes[k][fname][outputs[0]].attrs.keys():
                    if k.split('_')[0] == 'kwargs':
                        short_key = '_'.join(key.split('_')[1:])
                        kwargs[short_key] = processes[k][fname][outputs[0]].attrs[k]

                inputs = []
                for source in processes[k][fname][outputs[0]].attrs['source']:
                    inputs.append(f[source][()])

                results = func(*inputs, **kwargs)
                if type(results) != tuple:
                    results = tuple([results])

                for i in range(len(outputs)):
                    tmpattrs = {}
                    for k2 in processes[k][fname][outputs[i]].attrs.keys():
                        tmpattrs[k2] = processes[k][fname][outputs[i]].attrs[k2]

                    del (processes[k][fname][outputs[i]])
                    if processes[k][fname].attrs.__contains__('compressed'):
                        processes[k][fname].attrs.__delitem__('compressed')
                    processes[k][fname].create_dataset(outputs[i], data=results[i])
                    for k3 in tmpattrs.keys():
                        val = tmpattrs[k3]
                        processes[k][fname][outputs[i]].attrs.__setitem__(k3, val)