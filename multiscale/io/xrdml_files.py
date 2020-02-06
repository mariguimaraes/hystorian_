try:
    import xrdtools

    xrdtools_bool = True
except:
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
    counts = contents.split('<counts unit="counts">')[-1].split('</counts>')[0].split()
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
