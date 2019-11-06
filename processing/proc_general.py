import h5py
import numpy as np
from . import proc_tools as pt

def negative_(filename, data_folder='datasets', selection = None, selection_depth = 0):
    path_lists = pt.initialise_process(filename, 'negative')
    with h5py.File(filename, "a") as f:
        for path in path_lists[:]:
            neg = -np.array(f[path[0]])
            pt.generic_write(f, neg, path)