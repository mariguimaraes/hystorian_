import h5py
import csv
import numpy as np


def csv2hdf5(filename, filepath=None):
    if filepath is not None:
        file_base = filepath.split('.')[0]
    else:
        file_base = filename.split('.')[0]
    with open(filename) as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        data = list(data)
        header = data[0]
        data = data[1:]
        np_data = np.array(data).astype('S')

    with h5py.File(filename.split('.')[0] + ".hdf5", "w") as f:
        metadatagrp = f.create_group("metadata")
        metadatagrp.create_dataset(file_base, data=str(header))
        datagrp = f.create_group("datasets/" + file_base)
        if filepath is not None:
            datagrp.attrs.__setattr__('type', filepath.split('.')[-1])
        else:
            datagrp.attrs.__setattr__('type', filename.split('.')[-1])
        f.create_group("process")


        dataset = datagrp.create_dataset(file_base, data=np_data)
        dataset.attrs['name'] = file_base
        dataset.attrs['shape'] = np_data.shape
        dataset.attrs['header'] = header

    print('file successfully converted')
