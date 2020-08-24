try:
    import xrdtools
    xrdtools_bool = True
except ImportError:
    xrdtools_bool = False
import h5py
import re
import numpy as np


# ==========================================
# XRDML conversion

def xrdml2hdf5(filename):
    if not xrdtools_bool:
        print('Please download the xrdtools package if you want to use this function')
        return

    with open(filename, 'r') as f:
        contents = f.read()
        
    with h5py.File(filename.split('.')[0] + ".hdf5", "w") as f:
        param_names = ['counts', '2theta', 'omega', 'phi', 'chi', 'x', 'y']
        param_headers = ['',
                         '<positions axis="2Theta" unit="deg">',
                         '<positions axis="Omega" unit="deg">',
                         '<positions axis="Phi" unit="deg">',
                         '<positions axis="Chi" unit="deg">',
                         '<positions axis="X" unit="mm">',
                         '<positions axis="Z" unit="mm">']

        scans = contents.split('<dataPoints>')[1:]
        
        typegrp = f.create_group("type")
        typegrp.create_dataset(filename.split('.')[0], data=filename.split('.')[-1])

        metadatagrp = f.create_group("metadata")
        metadatagrp.create_dataset(filename.split('.')[0], data=contents)

        f.create_group("process")

        datagrp = f.create_group("datasets/" + filename.split('.')[0])
        
        for i in range(len(param_names)):
            name = param_names[i]
            all_data = []
            for scan in scans:
                if i == 0:
                    counts = scan.split('"counts">')[-1].split('</')[0].split()
                    val_list = list(map(float, counts))
                    step_num = len(val_list)
                if i != 0:
                    data_range = scan.split(param_headers[i])[1].split('</positions>')[0]
                    if len(data_range.split()) == 1:
                        val = float(re.findall(r"[-+]?\d*\.\d+|\d+", data_range.split()[0])[0])
                        val_list = np.full(step_num, val)
                    elif len(data_range.split()) == 2:
                        first_val = float(re.findall(r"[-+]?\d*\.\d+|\d+", data_range.split()[0])[0])
                        last_val = float(re.findall(r"[-+]?\d*\.\d+|\d+", data_range.split()[1])[0])
                        val_list = np.arange(first_val, last_val, (last_val - first_val) / step_num)
                all_data.append(val_list)
            all_data = np.array(all_data).T
            datagrp.create_dataset(name, data=all_data)
            datagrp[name].attrs['shape'] = np.shape(all_data)
            datagrp[name].attrs['name'] = 'counts'
            datagrp[name].attrs['size'] = 0
            datagrp[name].attrs['offset'] = 0
            datagrp[name].attrs['unit'] = 'au'
            if name == 'x' or name == 'y' or name == 'z':
                datagrp[name].attrs['unit'] = 'mm'
            elif name == 'counts':
                datagrp[name].attrs['unit'] = 'au'
            else:
                datagrp[name].attrs['unit'] = 'deg'

    print("File successfully converted")

    
def oldxrdml2hdf5(filename):
    if not xrdtools_bool:
        print('Please download the xrdtools package if you want to use this function')
        return

    with open(filename, 'r') as f:
        contents = f.read()
        
    counts = contents.split('"counts">')[-1].split('</')[0].split()
    cnts = list(map(float, counts))

    params = {'2theta': contents.split('<positions axis="2Theta" unit="deg">')[-1].split('</positions>')[0],
              'omega': contents.split('<positions axis="Omega" unit="deg">')[-1].split('</positions>')[0],
              'phi': contents.split('<positions axis="Phi" unit="deg">')[-1].split('</positions>')[0],
              'chi': contents.split('<positions axis="Chi" unit="deg">')[-1].split('</positions>')[0],
              'x': contents.split('<positions axis="X" unit="mm">')[-1].split('</positions>')[0],
              'y': contents.split('<positions axis="Z" unit="mm">')[-1].split('</positions>')[0]}

    angles = []
    for i, k in enumerate(params):
        if len(params[k].split()) == 1:
            val = float(re.findall(r"[-+]?\d*\.\d+|\d+", params[k].split()[0])[0])
        elif len(params[k].split()) == 2:
            first_val = float(re.findall(r"[-+]?\d*\.\d+|\d+", params[k].split()[0])[0])
            last_val = float(re.findall(r"[-+]?\d*\.\d+|\d+", params[k].split()[1])[0])

    with h5py.File(filename.split('.')[0] + ".hdf5", "w") as f:
        typegrp = f.create_group("type")
        typegrp.create_dataset(filename.split('.')[0], data=filename.split('.')[-1])

        metadatagrp = f.create_group("metadata")
        metadatagrp.create_dataset(filename.split('.')[0], data=contents)

        f.create_group("process")

        datagrp = f.create_group("datasets/" + filename.split('.')[0])
        datagrp.create_dataset("counts", data=cnts)
        datagrp['counts'].attrs['name'] = 'counts'
        datagrp['counts'].attrs['shape'] = len(cnts)
        datagrp['counts'].attrs['size'] = 0
        datagrp['counts'].attrs['offset'] = 0
        datagrp['counts'].attrs['unit'] = 'au'

        for i, k in enumerate(params):
            if len(params[k].split()) == 1:
                val = float(re.findall(r"[-+]?\d*\.\d+|\d+", params[k].split()[0])[0])
                val_list = np.full(len(cnts), val)
                datagrp.create_dataset(k, data=val_list)
                datagrp[k].attrs['shape'] = len(cnts)

            elif len(params[k].split()) == 2:
                first_val = float(re.findall(r"[-+]?\d*\.\d+|\d+", params[k].split()[0])[0])
                last_val = float(re.findall(r"[-+]?\d*\.\d+|\d+", params[k].split()[1])[0])
                val = np.arange(first_val, last_val, (last_val - first_val) / len(cnts))
                datagrp.create_dataset(k, data=val)
                datagrp[k].attrs['shape'] = len(cnts)

            # datagrp.create_dataset(k, data=val)
            datagrp[k].attrs['name'] = k
            datagrp[k].attrs['size'] = 0
            datagrp[k].attrs['offset'] = 0
            if k == 'x' or k == 'y' or k == 'z':
                datagrp[k].attrs['unit'] = 'mm'
            else:
                datagrp[k].attrs['unit'] = 'deg'

    print("File successfully converted")
