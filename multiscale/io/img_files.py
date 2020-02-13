import numpy as np
import h5py
from PIL import Image

def image2hdf5(filename):
    img = Image.open(filename)
    arr = np.array(img)

    with h5py.File(filename.split('.')[0] + ".hdf5", "w") as f:
        typegrp = f.create_group("type")
        typegrp.create_dataset(filename.split('.')[0], data=filename.split('.')[-1])

        metadatagrp = f.create_group("metadata")
        metadatagrp.create_dataset(filename.split('.')[0], data='no metadata')

        f.create_group("process")

        datagrp = f.create_group("datasets/" + filename.split('.')[0])
        keys = ['red', 'green', 'blue']
        for indx, key in enumerate(keys):
            datagrp.create_dataset(key, data=arr[:,:,indx])
            datagrp[key].attrs['name'] = key + ' channel'
            datagrp[key].attrs['shape'] = arr[:,:,indx].shape
            datagrp[key].attrs['size'] = (len(arr[:,:,indx]), len(arr[:,:,indx][0]))
            datagrp[key].attrs['offset'] = (0, 0)
