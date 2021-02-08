import numpy as np
import h5py

def dat2hdf5(filename, filepath=None, params=None):
    data = np.fromfile(filename,sep=" ")
    try:
        data = np.reshape(data, (int(np.sqrt(len(data))), int(np.sqrt(len(data)))))
    except ValueError:
        print("Shape of the data is not a square, keep it 1D")

    if params is not None:
        contents = np.fromfile(params, sep=" ")
        try:
            contents = np.reshape(contents, (3, int(len(contents)/3)))
        except ValueError:
            print("Shape is not 3xN, keep it 1D")




    with h5py.File(filename.split('.')[0] + ".hdf5", "w") as f:
        metadatagrp = f.create_group("metadata")
        f.create_group("process")

        if filepath is not None:
            if params is not None:
                metadatagrp.create_dataset(filepath.split('.')[0], data=contents)
            datagrp = f.create_group("datasets/" + filepath.split('.')[0])
            datagrp.attrs.__setattr__('type', filepath.split('.')[-1])

        else:
            if params is not None:
                metadatagrp.create_dataset(filename.split('.')[0], data=contents)
            datagrp = f.create_group("datasets/" + filename.split('.')[0])
            datagrp.attrs.__setattr__('type', filename.split('.')[-1])

        datagrp.create_dataset("SHG", data=data)