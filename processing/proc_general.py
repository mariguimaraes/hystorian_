import h5py
import numpy as np
from . import proc_tools as pt

#FUNCTION negative_
## Processes an array determine negatives of all values
## Trivial sample function to show how to use proc_tools
#INPUTS:
## filename: name of hdf5 file containing data
## data_folder: folder searched for inputs. eg. 'datasets', or 'process/negative'
## selection: determines the name of folders or files to be used. Can be None (selects all), a string, or a list of strings
## selection_depth: determines what level at which to look at a selection.
#OUTPUTS:
## null

def negative_(filename, data_folder='datasets', selection = None, selection_depth = 0):
    # Trivial sample function to show how to use proc_tools
    # Processes an array determine negatives of all values
    path_lists = pt.initialise_process(filename, 'negative', data_folder = data_folder, selection = selection, selection_depth = selection_depth)
    with h5py.File(filename, "a") as f:
        for path in path_lists[:]:
            neg = -np.array(f[path[0]])
            pt.generic_write(f, neg, path)