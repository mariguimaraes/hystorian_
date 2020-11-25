import h5py
import importlib
from multiscale import processing
import ast
import sys


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
            for key in f['process/001-sum_array/kpfm_5V_0000/sum1'].attrs.keys():
                if k.split('_')[0] == 'kwargs':
                    short_key = '_'.join(key.split('_')[1:])
                    kwargs[short_key] = processes[k][fname][outputs[0]].attrs[k]

            inputs = []
            for source in processes[k][fname][outputs[0]].attrs['source']:
                inputs.append(f[source][()])
            try:
                results = func(*inputs, **kwargs)
                for i in range(len(outputs)):
                    if (results[i] - processes[k][fname][outputs[i]][()] < error_threshold).all():
                        print('Result is not identical, not compressing')
                        identical = False
                        break
            except Exception as e:
                print(e.__doc__)
                print(e.message)
                identical = False

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

    processing.core.deallocate_hdf5_memory(file, verify=False)


