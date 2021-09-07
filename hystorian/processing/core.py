# 234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890

import h5py
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import hystorian.io.read_file as read_file
import fnmatch
import inspect
import sys
import time
from glob import glob
import types
import inspect

def m_apply(filename, function, in_paths, output_names=None, folder_names=None,
            use_attrs=None, prop_attrs=None, increment_proc=True, process_folder='process',
            **kwargs):
    """
    Take any function and handles the inputs of the function by looking into the hdf5 file
    Also write the output of the function into the hdf5.

    Parameters
    ----------
    filename: str
        name of the hdf5 file where the datas are stored
    function: function
        any function with input and outputs being array-like
    in_paths: list of str, or str
        list of the args for the function, usually the datasets
    output_names: list of str, or str, optional
        list of names for the outputs of the fonction
    folder_names: str, optional
        name of the folder for the output to be saved into the hdf5
        (default: second element of the first in_paths element)
    use_attrs: list of str, or str, optional
        list of attributes to be passed to the function as kwargs. (default: None)
    prop_attrs: list of str, or str, optional
        Propagate a list of attributes to the processed data. (default: None)
    increment_proc: bool, optional
        increment the process number in the hdf5 file. This should be kept to true, except during experimentation
        since setting it to false will overwrite processed datas and therefore lose history of the processing
        (default: True)
    process_folder, str, optional
        name of the folder in the hdf5 to save the output into. (default: 'process')
        This should stay 'process' almost all the times.
    **kwargs :
        any kwargs that the function passed to m_apply requires.

    Returns
    -------
        output of the passed function
    """
    # Convert in_paths to a list if not already
    if type(in_paths) != list:
        in_paths = [in_paths]
    
    # Check if paths needs to be searched, and if so, do so
    search = False
    for path in in_paths:
        if type(path) == list:
            for path in path:
                if '*' in path:
                    search = True
                    break
            if search:
                break
        else:
            if '*' in path:
                search = True
                break
    if search:
        in_paths = path_search(filename, in_paths)
        in_paths = list(map(list, zip(*in_paths)))[0]
        
    # Guess output_names (aka channel names) if not given
    if output_names is None:
        output_names = in_paths[0].rsplit('/', 1)[1]

    # Guess folder_names (aka sample names) if not given
    if folder_names is None:
        folder_names = in_paths[0].rsplit('/', 2)[1]

    # Convert output_names to list if not already
    if type(output_names) != list:
        output_names = [output_names]

    # Convert prop_attrs to list if it exists, but not already a list
    if prop_attrs is not None:
        if type(prop_attrs) != list:
            prop_attrs = [prop_attrs]

    # Convert use_attrs to list if it exists, but not already a list
    if use_attrs is not None:
        if type(use_attrs) != list:
            use_attrs = [use_attrs]

    # Convert file to hdf5 if not already
    if filename.split('.')[-1] != 'hdf5':
        if os.path.isfile(filename.split('.')[0] + '.hdf5'):
            filename = filename.split('.')[0] + '.hdf5'
        else:
            try:
                read_file.tohdf5(filename)
                filename = filename.split('.')[0] + '.hdf5'
                print('The file does not have an hdf5 extension. It has been converted.')
            except:
                print('The given filename does not have an hdf5 extension, and it was not possible''to convert it. '
                      'Please use an hdf5 file with m_apply')

    # Open hdf5 file to extract data, attributes, and run function
    data_list = []
    prop_attr_keys = []
    prop_attr_vals = []
    use_attr_keys = []
    use_attr_vals = []
    with h5py.File(filename, 'r') as f:
        for path in in_paths:
            data_list.append(np.array(f[path]))
            if prop_attrs is not None:
                for prop_attr in prop_attrs:
                    if (prop_attr == 'scale_m_per_px'):
                        if 'scale (m/px)' in f[path].attrs:
                            prop_attr_keys.append(prop_attr)
                            prop_attr_vals.append(f[path].attrs['scale (m/px)'])
                    if (prop_attr not in prop_attr_keys) and (prop_attr in f[path].attrs):
                        prop_attr_keys.append(prop_attr)
                        prop_attr_vals.append(f[path].attrs[prop_attr])
            if use_attrs is not None:
                for use_attr in use_attrs:
                    if (use_attr not in use_attr_keys) and (use_attr in f[path].attrs):
                        use_attr_keys.append(use_attr)
                        use_attr_vals.append(f[path].attrs[use_attr])
                use_attr_dict = {}
                for key_num in range(len(use_attr_keys)):
                    use_attr_dict['source_' + use_attr_keys[key_num]] = use_attr_vals[key_num]
                kwargs.update(use_attr_dict)

        result = function(*data_list, **kwargs)

    # End function if no result is calculated
    if isinstance(result, type(None)):  # type(result) == type(None):
        return None

    # Convert result to tuple if not already
    if type(result) != tuple:
        result = tuple([result])

    # Open hdf5 file to write new data, attributes
    with h5py.File(filename, 'a') as f:
        num_proc = len(f[process_folder].keys())
        if increment_proc:
            num_proc = num_proc + 1
            out_folder_location = (process_folder + '/' + str(num_proc).zfill(3) + '-' +
                                   function.__name__ + '/' + folder_names)
        else:
            out_folder_location = (process_folder + '/' + str(num_proc).zfill(3) + '-' +
                                   function.__name__ + '/' + folder_names)
            if out_folder_location.split('/')[1] not in f['process'].keys():
                out_folder_location = (process_folder + '/' + str(num_proc + 1).zfill(3) + '-' +
                                       function.__name__ + '/' + folder_names)

        fproc = f.require_group(out_folder_location)

        if len(output_names) == len(result):
            for i in range(len(output_names)):
                name = output_names[i]
                data = result[i]
                if type(data) == dict:
                    if 'hdf5_dict' in data:
                        dataset = create_dataset_from_dict(f[out_folder_location], name, data)
                        if prop_attrs is not None:
                            dataset = propagate_attrs(dataset, prop_attr_keys, prop_attr_vals)
                    else:
                        dataset = create_dataset_with_name_check(f[out_folder_location], name, data)
                        if prop_attrs is not None:
                            dataset = propagate_attrs(dataset, prop_attr_keys, prop_attr_vals)
                else:
                    dataset = create_dataset_with_name_check(f[out_folder_location], name, data)
                    if prop_attrs is not None:
                        dataset = propagate_attrs(dataset, prop_attr_keys, prop_attr_vals)
                write_generic_attributes(fproc[name], out_folder_location + '/', in_paths, name, function)
                write_kwargs_as_attributes(fproc[name], function, kwargs, first_kwarg=len(in_paths))
        else:
            print('Error: Unequal amount of outputs and output names')
    return result


#   FUNCTION l_apply
# Runs m_apply multiple times successively, intended to operate on an entire process or dataset
# folder
#   INPUTS:
# filename : name of the hdf5 file where the datas are stored
# function : Custom function that you want to call
# all_input_criteria : Regex expression to describe the inputs searched for. Can be composed as a
#     list of a list of strings, with extra list parenthesis automatically generated. Eg:
#         'process*Trace1*' would pass to m_apply all files that contain 'process*Trace1*'.
#         ['process*Trace1*'] as above
#         [['process*Trace1*']] as above
#         [['process*Trace1*', 'process*Trace2*']] would pass to m_apply all files that contain 
#             'process*Trace1*' and 'process*Trace2*' in a single list.
#         [['process*Trace1*'], ['process*Trace2*']] would pass to m_apply all files that contain 
#             'process*Trace1*' and 'process*Trace2*' in two different lists; and thus will operate
#             differently on each of these lists.
# repeat (default: None): determines what to do if path_lists generated are of different lengths.
#     None: Default, no special action is taken, and extra entries are removed. ie, given lists
#         IJKL and AB, IJKL -> IJ.
#     'alt': The shorter lists of path names are repeated to be equal in length to the longest list.
#         ie, given IJKL and AB, AB -> ABAB
#     'block': Each entry of the shorter list of path names is repeated to be equal in length to the
#         longest list. ie, given IJKL and AB, AB -> AABB.
# **kwargs : All the non-data inputs to give to the function
#    OUTPUTS:
# null
#    TO DO:
# Can we force a None to be passed?

def l_apply(filename, function, all_input_criteria, repeat=None, **kwargs):
    all_in_path_list = path_search(filename, all_input_criteria, repeat)
    all_in_path_list = list(map(list, zip(*all_in_path_list)))
    increment_proc = True
    start_time = time.time()
    for path_num in range(len(all_in_path_list)):
        m_apply(filename, function, all_in_path_list[path_num], increment_proc=increment_proc,
                **kwargs)
        progress_report(path_num + 1, len(all_in_path_list), start_time, function.__name__,
                        all_in_path_list[path_num])
        increment_proc = False


def print_tree(filename, root=''):
    '''
    Prints the file structure of a .hdf5 file.

    Parameters
    ----------
    filename : name of file to be examined
    root : optional, allows showing only a subset of the tree (default : '')

    Returns
    -------
    null
    '''

    def print_branches(name):
        space = '    '
        branch = '│   '
        tee = '├── '
        last = '└── '

        shift = ''
        curr_name_segments = (name.split('/'))
        last_name_segments = []
        for i in range((len(curr_name_segments))):
            if i == 0:
                if not root:
                    last_name_segments.append(list(f.keys())[-1])
                else:
                    last_name_segments.append(list(f[root].keys())[-1])
            else:
                if not root:
                    last_name_segments.append(list(f[name.rsplit('/', len(curr_name_segments) - i)[0]])[-1])
                else:
                    last_name_segments.append(list(f[root][name.rsplit('/', len(curr_name_segments) - i)[0]])[-1])
        for i in range((len(curr_name_segments))):
            curr_name = curr_name_segments[i]
            last_name = last_name_segments[i]
            if i != len(curr_name_segments) - 1:
                if curr_name != last_name:
                    shift += branch
                else:
                    shift += space
            else:
                if curr_name != last_name:
                    shift += tee
                else:
                    shift += last
        item_name = name.split("/")[-1]
        print(shift + item_name)

    with h5py.File(filename, "r") as f:
        if root:
            print(filename + '[' + root + ']')
            f[root].visit(print_branches)
        else:
            print(filename)
            f.visit(print_branches)


def list_processes(filename, process=''):
    '''
    Returns a list of all processes in a named .hdf5 file.

    Parameters
    ----------
    filename : name of file to be searched
    process : optional, shortens the list of processes using regex (default : '')

    Returns
    -------
    proclst : a list of all processes
    '''

    procs = np.array(pt.path_search(filename, 'process*' + process + '*')).ravel()
    if len(procs) == 0:
        return None
    proclst = [p.split('/')[1].split('/')[0] for p in procs]
    proclst = sorted(list(set(proclst)))
    return proclst


def last_process(filename, process=''):
    '''
    Returns a the filename of the last process as a string

    Parameters
    ----------
    filename : name of file to be searched
    process : optional, further qualifies the last process by a regex condition (default : '')

    Returns
    -------
    final process filling the regex condition (if any)
    '''
    list_of_processes = list(list_processes(filename, process=''))
    return (list_of_processes[-1])


def remove_process(filename, process):
    '''
    Clears a named process from a named .hdf5 file. Note that the data is not de-allocated
    in computer memory; use core.deallocate_hdf5_memory to regain memory.

    Parameters
    ----------
    filename : name of file to be altered
    process : string containing name of process to be removed

    Returns
    -------
    null
    '''
    process_path = 'process/' + process
    with h5py.File(filename, "a") as f:
        if process_path in f.keys():
            del f[process_path]
            print(process_path + ' has been removed')
        else:
            print(process_path + ' not found')

#   FUNCTION path_search
# Uses regex expressions to search for all paths. Useful when writing complicated custom functions
# that cannot use m_apply or l_apply
#   INPUTS:
# filename : name of the hdf5 file where the datas are stored
# all_input_criteria : Regex expression to describe the inputs searched for. Can be composed as a
#     list of a list of strings, with extra list parenthesis automatically generated. Eg:
#         'process*Trace1*' would pass to m_apply all files that contain 'process*Trace1*'.
#         ['process*Trace1*'] as above
#         [['process*Trace1*']] as above
#         [['process*Trace1*', 'process*Trace2*']] would pass to m_apply all files that contain 
#             'process*Trace1*' and 'process*Trace2*' in a single list.
#         [['process*Trace1*'], ['process*Trace2*']] would pass to m_apply all files that contain 
#             'process*Trace1*' and 'process*Trace2*' in two different lists; and thus will operate
#             differently on each of these lists.
# repeat (default: None): determines what to do if path_lists generated are of different lengths.
#     None: Default, no special action is taken, and extra entries are removed. ie, given lists
#         IJKL and AB, IJKL -> IJ.
#     'alt': The shorter lists of path names are repeated to be equal in length to the longest list.
#         ie, given IJKL and AB, AB -> ABAB
#     'block': Each entry of the shorter list of path names is repeated to be equal in length to the
#         longest list. ie, given IJKL and AB, AB -> AABB.
#    OUTPUTS:
# all_in_path_list: list of paths (paths are strings)


def path_search(filename, all_input_criteria, repeat=None):
    if type(all_input_criteria) != list:
        all_input_criteria = [all_input_criteria]
    if type(all_input_criteria[0]) != list:
        all_input_criteria = [all_input_criteria]

    with h5py.File(filename, 'r') as f:
        all_path_list = find_paths_of_all_subgroups(f, 'datasets')
        all_path_list.extend(find_paths_of_all_subgroups(f, 'process'))

        all_in_path_list = []
        list_lengths = []
        for each_data_type in all_input_criteria:
            in_path_list = []
            for each_criteria in each_data_type:
                for path in all_path_list:
                    if fnmatch.fnmatch(path, each_criteria):
                        in_path_list.append(path)
            all_in_path_list.append(in_path_list)
            list_lengths.append(len(in_path_list))
        if len(list_lengths) == 1:
            if list_lengths[0] == 0:
                print('No Input Datafiles found!')
        else:
            if len(set(list_lengths)) != 1:
                if repeat is None:
                    print('Input lengths not equal, and repeat not set! Extra files will be' \
                          'omitted.')
                else:
                    largest_list_length = np.max(list_lengths)
                    list_multiples = []
                    for length in list_lengths:
                        if largest_list_length % length != 0:
                            print('At least one path list length is not a factor of the largest' \
                                  'path list length. Extra files will be omitted.')
                        list_multiples.append(largest_list_length // length)
                    if (repeat == 'block') or (repeat == 'b'):
                        for list_num in range(len(list_multiples)):
                            all_in_path_list[list_num] = np.repeat(all_in_path_list[list_num],
                                                                   list_multiples[list_num])
                    if (repeat == 'alt') or (repeat == 'a'):
                        for list_num in range(len(list_multiples)):
                            old_path_list = all_in_path_list[list_num]
                            new_path_list = []
                            for repeat_iter in range(list_multiples[list_num]):
                                new_path_list.extend(old_path_list)
                            all_in_path_list[list_num] = new_path_list
    return all_in_path_list


#   FUNCTION create_dataset_from_dict
# Subfunction used in m_apply. Converts the hdf5_dict output file into a dataset that is written to
# the hdf5 file, with all attributes encoded. Also checks if a folder of the same name is present,
# and if it should be written.
#   INPUTS:
# dataset: Path to the datasets in the hdf5 file that contain the input.
# name: name of the channel the results are written in.
# dict_data: the dict that contains the data written. This should include both the actual data
#     to be written, as well as additional attributes.
#   OUTPUTS:
# dataset: The dataset written to

def create_dataset_from_dict(dataset, name, dict_data):
    if name in dataset:
        print(str(name) + ' already exists in' + dataset.name + '.')
        response = input(' Would you like to overwrite the dataset? (y/n)')
        if response != 'y':
            return dataset
        else:
            dataset[name][...] = dict_data['data']
    else:
        dataset = dataset.create_dataset(name, data=dict_data['data'])
    for key, value in dict_data.items():
        if (key != 'hdf5_dict') and (key != 'data'):
            dataset.attrs[key] = value
    return dataset


#   FUNCTION create_dataset_with_name_check
# Subfunction used in m_apply. Writes the output, but also checks if a folder of the same name is
# present, and if it should be written.
#   INPUTS:
# dataset: Path to the datasets in the hdf5 file that contain the input.
# name: name of the channel the results are written in.
# dict_data: the dict that contains the data written. This should include both the actual data
#     to be written, as well as additional attributes.
#   OUTPUTS:
# dataset: The dataset written to

def create_dataset_with_name_check(dataset, name, data):
    if name in dataset:
        print(str(name) + ' already exists in ' + dataset.name + '.')
        response = input(' Would you like to overwrite the dataset? (y/n)')
        if response != 'y':
            return dataset
        else:
            dataset[name][...] = data
    else:
        dataset = dataset.create_dataset(name, data=data)
    return dataset


#   FUNCTION propagate_attrs
# Subfunction used in m_apply. Propagates attributes into a new dataset, with their keys and values
# given.
#   INPUTS:
# dataset: Path to the datasets in the hdf5 file that contain the input.
# prop_attr_keys: A list of strings that indicate the names of the actual attributes to be written.
# prop_attr_vals: A list that contains the values of the attributes to be propogated. Should be
#     ordered with prop_attr_keys
#   OUTPUTS:
# dataset: The dataset written to

def propagate_attrs(dataset, prop_attr_keys=[], prop_attr_vals=[]):
    for i_attr in range(len(prop_attr_keys)):
        dataset.attrs[prop_attr_keys[i_attr]] = prop_attr_vals[i_attr]
    return dataset


#   FUNCTION hdf5_dict
# Called in custom functions to create the hdf5_dict that is used by m_apply to write the dataset and
# associated attributes.
#   INPUTS:
# dataset_location: Path to the datasets in the hdf5 file that contain the input.
# name: name of the channel the results are written in.
# dict_data: the dict that contains the data written. This should include both the actual data
#     to be written, as well as additional attributes.
# prop_attrs: The original list of strings that indicate the attributes to be propagated. Used to
#     check if attribute propagation should occur.
# prop_attr_keys: A list of strings that indicate the names of the actual attributes to be written.
# prop_attr_vals: A list that contains the values of the attributes to be propogated. Should be
#     ordered with prop_attr_keys
#   OUTPUTS:
# dataset: The dataset written to

def hdf5_dict(data, **kwargs):
    data_dict = {
        'hdf5_dict': True,
        'data': data
    }
    data_dict.update(kwargs)
    return data_dict


#   FUNCTION write_generic_attributes
# Writes necessary and generic attributes to a dataset. This includes the dataset shape, its name,
# operation name and number, time of writing, and source file(s).
#   INPUTS:
# dataset: the dataset the attributes are written to
# out_folder_location: location of the dataset
# in_paths: the paths to the source files
# output_name: the name of the dataset
#   OUTPUTS
# null

def write_generic_attributes(dataset, out_folder_location, in_paths, output_name, function):
    if type(in_paths) != list:
        in_paths = [in_paths]
    operation_name = out_folder_location.split('/')[1]

    dataset.attrs['path'] = out_folder_location + output_name
    dataset.attrs['shape'] = dataset.shape
    dataset.attrs['name'] = output_name
    #dataset.attrs['operation name'] = operation_name.split('-')[1]
    if function.__module__ is None:
        dataset.attrs['operation name'] = 'None.' + function.__name__
    else:
        dataset.attrs['operation name'] = function.__module__ + '.' + function.__name__
    if function.__module__ == '__main__':
        dataset.attrs['function code'] = inspect.getsource(function)
    dataset.attrs['operation number'] = operation_name.split('-')[0]
    dataset.attrs['time'] = str(datetime.now())
    dataset.attrs['source'] = in_paths

    
#   FUNCTION write_kwargs_as_attributes
# Writes all other arguments as attributes to a datset.
#   INPUTS:
# dataset: the dataset the attributes are written to
# func: the function from which attributes are pulled
# all_variables: all variables from the function call. To call properly, set as locals()
# first_kwarg (default: 1): First kwarg that is to be written in
#   OUTPUTS
# null
    
def write_kwargs_as_attributes(dataset, func, all_variables, first_kwarg=1):
    if isinstance(func, types.BuiltinFunctionType):
        dataset.attrs['BuiltinFunctionType'] = True
    else:
        varargs = inspect.getfullargspec(func).varargs
        signature = inspect.signature(func).parameters
        first_arg = list(signature.keys())[0]
        if varargs == first_arg:
            first_kwarg=1
        var_names = list(signature.keys())[first_kwarg:]
        for key in var_names:
            if key in all_variables:
                value = all_variables[key]
            else:
                value = signature[key].default
            if callable(value):
                value = value.__name__
            elif value is None:
                value = 'None'
            try:
                dataset.attrs['kwargs_' + key] = value
            except RuntimeError:
                print('Attribute was not able to be saved, probably because the attribute'
                      'is too large')
                dataset.attrs['kwargs_' + key] = 'None'
    

#   FUNCTION progress_report
# Prints progression of a process run in several stages
#   INPUTS:
# processes_complete: number of processes that have currently been run
# processes_total: total number of processes that have been and will be run
# start_time: time at which the process began
# process_name (default: 'undefined_process'): name of the process run
# identifier (default: '[unidentified_sample]'): name of the particular iteration of the process
# clear (default: True): Decides whether to overwrite existing print statements
#   OUTPUTS
# null

def progress_report(processes_complete, processes_total, start_time=None,
                    process_name='undefined_process', identifier='[unidentified_sample]',
                    clear=True):
    if processes_complete != processes_total:
        if start_time is not None:
            time_remaining = round(((processes_total - processes_complete) / processes_complete) *
                                   (time.time() - start_time))
            str_progress = (process_name + ': ' + str(processes_complete) + ' of ' +
                            str(processes_total) + ' Complete. ' + str(time_remaining) +
                            's remaining. ' + str(identifier))
        else:
            str_progress = (process_name + ': ' + str(processes_complete) + ' of ' +
                            str(processes_total) + ' Complete. Unknown time remaining. ' +
                            str(identifier))
        if clear:
            print(str_progress + ' ' * len(str_progress), sep=' ', end='\r', file=sys.stdout,
                  flush=False)
        else:
            print(str_progress)
    if processes_complete == processes_total:
        str_final = (process_name + ' complete at ' + time.strftime('%H:%M', time.localtime()) + '! '
                     + str(processes_complete) + ' processes performed in '
                     + str(round(time.time() - start_time)) + 's.')
        print(str_final + ' ' * 1000)


#   FUNCTION intermediate_plot
# Debugging tool that shows basic images in the kernel. The function checks if the keyword 
# 'condition' is in the list 'plotlist'. If so, the data is plotted and text printed. Alternatively,
# if force_plot is True, the plot is shown and text printed with no other checks. Otherwise, the
# function does nothing
#   INPUTS:
# data: data to be plotted
# condition (default: ''): a string checked to be in 'plotlist'
# plotlist (default: []): a list of strings that may contain 'condition'
# text (default: 'Intermediate Plot'): text printed prior to showing the plot.
# force_plot (default: False): if set to True, all other conditions are bypassed and the image is 
#     plotted.
#   OUTPUTS
# null

def intermediate_plot(data, condition='', plotlist=[], text='Intermediate Plot', force_plot=False):
    if force_plot:
        print(text)
        plt.imshow(data)
        plt.show()
        plt.close()
    elif type(plotlist) == list:
        if condition in plotlist:
            print(text)
            plt.imshow(data)
            plt.show()
            plt.close()


def find_paths_of_all_subgroups(f, current_path=''):
    """
    Recursively determines list of paths for all datafiles in current_path, as well as datafiles in
    all subfolders (and sub-subfolders and...) of current path. If no path given, will find all 
    subfolders in entire file.
      
    Parameters
    ----------
    f : open file
        open hdf5 file
    current_path : string
        current group searched
    
    Returns
    ------
    path_list : list
        list of paths to datafiles
    """
    path_list = []
    if current_path == '':
        curr_group = f
    else:
        curr_group = f[current_path]
    for sub_group in curr_group:
        if isinstance(f[current_path + '/' + sub_group], h5py.Group):
            path_list.extend(find_paths_of_all_subgroups(f, current_path + '/' + sub_group))
        elif isinstance(f[current_path + '/' + sub_group], h5py.Dataset):
            path_list.append(current_path + '/' + sub_group)
    return path_list


#   FUNCTION rms
# Return the root mean square of an array
#   INPUTS:
# array: an array of any dimension
#   OUTPUTS
# RMS: The root mean square of the array
# Skewness: measure of the asymmetry of the probability distribution of a real-valued random variable
#    about its mean (Wiki)
# excess kurtosis : higher kurtosis corresponds to greater extremity of deviations (or outliers)
#    (Wiki)

def rms(array):
    print(array.size)
    mu2 = 1 / array.size * np.sum((array - np.mean(array)) ** 2)
    mu3 = 1 / array.size * np.sum((array - np.mean(array)) ** 3)
    mu4 = 1 / array.size * np.sum((array - np.mean(array)) ** 4)

    return np.sqrt(mu2), mu3 / (mu2 ** (3.0 / 2.0)), mu4 / mu2 ** 2 - 3


def deallocate_hdf5_memory(filename, verify=True):
    """
    By default, .hdf5 files do not deallocate memory after component datasets or groups are deleted.
    This function overcomes this issue by 'repacking' the data in a .hdf5 file into a new .hdf5 file.
    The original file is then deleted and replaced with the new file.
      
    Parameters
    ----------
    filename : string
        the hdf5 file to have memory deallocated
    verify : bool
        If true, asks for the verification request.
    
    Returns
    ------
        None
    """
    if verify:
        print('WARNING: The basic process of this function involves:')
        print('    1) Copying all components of a .hdf5 file into another .hdf5 file,')
        print('    2) Deleting the original file, and')
        print('    3) Renaming the new file to the original file.')
        print('As a result of this deletion-rename process, there is a possibility')
        print('that files may be lost. It is -strongly- recommended to back up all')
        print('files before running this function.\n')
        print('Once all files are backed up, type \'Yes\' to continue.')
        response = input()
        if response != 'Yes':
            print('Invalid Input; Function aborted.')
            return
    if '_CleanCopy.hdf5' in filename:
        raise ValueError('Filename ending in _CleanCopy.hdf5 found. Please remove from folder.')
    copy_filename = filename.split('.hdf5')[0] + '_CleanCopy.hdf5'
    with h5py.File(filename, 'a') as f1:
        with h5py.File(copy_filename, 'a') as f2:
            for group in f1:
                f1.copy(group, f2)
    os.remove(filename)
    os.rename(copy_filename, filename)


def deallocate_hdf5_memory_of_folder(folder_path, verify=True):
    """
    As above, but operates on an entire folder of .hdf5 files.
      
    Parameters
    ----------
    folder_path : string
        path to folder to have memory deallocated
    verify : bool
        If true, asks for the verification request.
    
    Returns
    ------
        None
    """
    os.chdir(folder_path)
    filelist = glob('*.hdf5')
    if verify:
        print('WARNING: The basic process of this function involves:')
        print('    1) Copying all components of a .hdf5 file into another .hdf5 file,')
        print('    2) Deleting the original file, and')
        print('    3) Renaming the new file to the original file.')
        print('As a result of this deletion-rename process, there is a possibility')
        print('that files may be lost. It is -strongly- recommended to back up all')
        print('files before running this function.\n')
        print('The following .hdf5 files will be repacked to reduce memory:')
        for filename in filelist:
            print('    ' + filename)
        print('')
        print('Once all files are backed up, type \'Yes\' to continue.')
        response = input()
        if response != 'Yes':
            print('Invalid Input; Function aborted.')
            return
    for filename in filelist:
        if '_CleanCopy.hdf5' in filename:
            raise ValueError('Filename ending in _CleanCopy.hdf5 found. Please remove from folder.')
    for filename in filelist:
        copy_filename = filename.split('.hdf5')[0] + '_CleanCopy.hdf5'
        with h5py.File(filename, 'a') as f1:
            with h5py.File(copy_filename, 'a') as f2:
                for group in f1:
                    f1.copy(group, f2)
        os.remove(filename)
        os.rename(copy_filename, filename)


    
def create_group_path(f, path):
    groups = path.split('/')
    curr_path = ''
    for curr_group in groups:
        if curr_path == '':
            if curr_group not in f.keys():
                f.create_group(curr_group)
        else:
            if curr_group not in f[curr_path].keys():
                f[curr_path].create_group(curr_group)
        curr_path = curr_path+'/'+curr_group
        

#####################################################################################################
#                                                                                                   #
#                    THESE FUNCTIONS ARE USEFUL FOR HARDCODING COMPLEX FUNCTIONS                    #
#                                                                                                   #
#####################################################################################################


#   FUNCTION find_output_folder_location
# Creates a list of paths to the locations of the output folder. These paths lead to the process
# folder, contain a number corresponding to the operation number (starting from 1), and contain the
# process name. The 'sample' folder can be passed manually, or inherited from the source folder.
#   INPUTS:
# filename or f: either the open datafile, or the filename of a .hdf5 file that can be opened.
# process_name: the name of the process folder
# folder_names: The 'sample' folder name as a string. Alternatively, the list of source folders
#     can instead be passed. The folder name then directly copies from these source folders.
# overwrite if same (default: False): if set to True, if this function was the last process run, the
#     last run will be overwritten and replaced with this. To be used sparingly, and only if
#     function parameters must be guessed and checked
#   OUTPUTS:
# out_folder_location_list: list of paths to output folders

def find_output_folder_location(filename_or_f, process_name, folder_names,
                                overwrite_if_same=False):
    out_folder_location_list = []
    if type(filename_or_f) == str:
        filename = filename_or_f
        with h5py.File(filename, 'a') as f:
            out_folder_location_list = find_output_folder_location(f, process_name, folder_names,
                                                                   overwrite_if_same)
    elif type(filename_or_f) == h5py._hl.files.File:
        f = filename_or_f
        operation_number = len(f['process']) + 1
        if overwrite_if_same:
            if str(operation_number - 1).zfill(3) + '-' + process_name in f['process'].keys():
                operation_number = operation_number - 1
        if (type(folder_names) != list) and (type(folder_names) != np.ndarray):
            folder_names = [folder_names]

        for folder in folder_names:
            if '/' in folder:
                folder_root, output_filename = folder.rsplit('/', 1)
                if folder_root.split('/', 1)[0] == 'datasets':
                    folder_centre = folder_root.split('/', 1)[1]
                elif folder_root.split('/', 1)[0] == 'process':
                    folder_centre = folder_root.split('/', 2)[2]
                else:
                    print('Error: folder_names should not contain a slash unless a path to either ' \
                          'datasets or process')
            else:
                folder_centre = folder
            out_folder_location = ('process/' + str(operation_number).zfill(3) + '-' + process_name
                                   + '/' + folder_centre + '/')
            out_folder_location_list.append(out_folder_location)
    return out_folder_location_list


#   FUNCTION write_output_f
# Writes the output to an open datafile
#   INPUTS:
# f: the open datafile
# data: the data to be written
# out_folder_location: location the dataset is written to
# in_paths: list of paths for the sources of the data
# function: the function that calls write_output_f
# all_variables: all variables from the function call. To call properly, set as locals()
# first_kwarg (default: 2): First kwarg that is to be written in
# output_name (default: None): the name of the datafile to be written. If left as None, the output
#     name is inherited from the name of the first entry of in_paths
#   OUTPUTS
# dataset: the directory to the dataset, such that f[dataset] would yield data

def write_output_f(f, data, out_folder_location, in_paths, function, all_variables, first_kwarg=2,
                   output_name=None):
    f.require_group(out_folder_location)
    if (type(in_paths) != list) and (type(in_paths) != np.ndarray):
        in_paths = [in_paths]
    if output_name is None:
        if (type(in_paths[0]) == str) or (type(in_paths[0]) == np.str_):
            output_name = in_paths[0].rsplit('/', 1)[1]
        else:
            output_name = in_paths[0][0].rsplit('/', 1)[1]
    try:
        # By default doesn't allow overwrite, so delete before writing
        del f[out_folder_location][output_name]
    except:
        pass
    f[out_folder_location].create_dataset(output_name, data=data)
    dataset = f[out_folder_location][output_name]
    write_generic_attributes(dataset, out_folder_location, in_paths, output_name, function)
    write_kwargs_as_attributes(dataset, function, all_variables, first_kwarg)
    return dataset


#   FUNCTION read_dataset
# Given the filename and input criteria, returns the dataset as an array
#   INPUTS:
# filename : name of the hdf5 file where the datas are stored
# all_input_criteria : Regex expression to describe the inputs searched for
#   OUTPUTS
# data: the requested dataset


def read_dataset(filename, all_input_criteria):
    all_in_path_list = path_search(filename, all_input_criteria)
    all_in_path_list = list(map(list, zip(*all_in_path_list)))
    data_list = []
    for path in all_in_path_list:
        with h5py.File(filename, 'r') as f:
            data_list.append(np.array(f[path[0]]))
    data = np.array(data_list)
    return data