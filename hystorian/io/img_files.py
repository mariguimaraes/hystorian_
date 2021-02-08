import numpy as np
import h5py
from PIL import Image


def image2hdf5(filename, filepath=None):
    img = Image.open(filename)
    arr = np.array(img)

    with h5py.File(filename.split('.')[0] + ".hdf5", "w") as f:
        metadatagrp = f.create_group("metadata")
        f.create_group("process")

        if filepath is not None:
            datagrp = f.create_group("datasets/" + filepath.split('.')[0])
            metadatagrp.create_dataset(filepath.split('.')[0], data='no metadata')
            datagrp.attrs.__setattr__('type', filepath.split('.')[-1])

        else:
            datagrp = f.create_group("datasets/" + filename.split('.')[0])
            metadatagrp.create_dataset(filename.split('.')[0], data='no metadata')
            datagrp.attrs.__setattr__('type', filename.split('.')[-1])


        keys = ['red', 'green', 'blue']
        for indx, key in enumerate(keys):
            datagrp.create_dataset(key, data=arr[:, :, indx])
            datagrp[key].attrs['name'] = key + ' channel'
            datagrp[key].attrs['shape'] = arr[:, :, indx].shape
            datagrp[key].attrs['size'] = (len(arr[:, :, indx]), len(arr[:, :, indx][0]))
            datagrp[key].attrs['offset'] = (0, 0)
