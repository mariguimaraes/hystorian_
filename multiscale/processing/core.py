# 234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890

import h5py
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import multiscale.io.read_file as read_file
import fnmatch
import sys
import time


#   FUNCTION m_apply
# Take any function and handles the inputs of the function by looking into the hdf5 file
# The input must be in the form datasets/name/channel or process/proc_name/name/channel
# Also write the output of the function into the hdf5. The name of the path can be given into 
# outputs_names
#   INPUTS:
# filename : name of the hdf5 file where the datas are stored
# function : Custom function that you want to call
# in_paths : Paths to the datasets in the hdf5 file that contain the inputs. Multiple inputs can
#     be given by using a list
# outputs_names (default: None): list of the names of the channels the results are written in.
#     By default, copies names from the first of the in_paths
# folder_names (default: None): list of the names of the folder containing results data channels.
#     By default, copies names from the first of the in_paths
# prop_attrs (default: None): string, or list of strings, that are the names of attributes that will
#     be copied from in_paths, into each output file. If the same attribute name is in multiple
#     in_paths, the first in_path with the attribute name will be copied from.
# increment_proc (default: True): determines whether to increment the process counter
# **kwargs : All the non-data inputs to give to the function
#    OUTPUTS:
# result: the datafile produced after running the custom function

def m_apply(filename, function, in_paths, output_names=None, folder_names = None,
            prop_attrs = None, increment_proc = True, **kwargs):
    
    #Convert in_paths to a list if not already
    if type(in_paths) != list:
        in_paths = [in_paths]
    
    #Guess output_names (aka channel names) if not given
    if output_names is None:
        output_names = in_paths[0].rsplit('/', 1)[1]
    
    #Guess folder_names (aka sample names) if not given
    if folder_names is None:
        folder_names = in_paths[0].rsplit('/', 2)[1]
    
    #Convert output_names to list if not already
    if type(output_names) != list:
        output_names = [output_names]
    
    #Convert prop_attrs to list if it exists, but not already a list
    if prop_attrs is not None:
        if type(prop_attrs) != list:
            prop_attrs = [prop_attrs]
    
    #Convert file to hdf5 if not already
    if filename.split('.')[-1] != 'hdf5':
        if os.path.isfile(filename.split('.')[0] + '.hdf5'):
            filename = filename.split('.')[0] + '.hdf5'
        else:
            try:
                read_file.tohdf5(filename)
                filename = filename.split('.')[0] + '.hdf5'
                print('The file does not have an hdf5 extension. It has been converted.')
            except:
                print('The given filename does not have an hdf5 extension, and it was not possible' \
                        'to convert it. Please use an hdf5 file with m_apply')
                
    #Open hdf5 file to extract data, attributes, and run function
    data_list = []
    prop_attr_keys = []
    prop_attr_vals = []
    with h5py.File(filename, 'r') as f:
        for path in in_paths:
            data_list.append(np.array(f[path]))
            if prop_attrs is not None:
                for prop_attr in prop_attrs:
                    if (prop_attr not in prop_attr_keys) and (prop_attr in f[path].attrs):
                        prop_attr_keys.append(prop_attr)
                        prop_attr_vals.append(f[path].attrs[prop_attr])
        result = function(*data_list, **kwargs)
    
    #End function if no result is calculated
    if isinstance(result, type(None)):  # type(result) == type(None):
        return None

    #Convert result to tuple if not already
    if type(result) != tuple:
        result = tuple([result])
    
    #Open hdf5 file to write new data, attributes
    with h5py.File(filename, 'a') as f:
        num_proc = len(f['process'].keys())
        if increment_proc:
            num_proc = num_proc + 1
        out_folder_location = ('process/' + str(num_proc).zfill(3) + '-' + function.__name__ + '/'
                               + folder_names)
        fproc = f.require_group(out_folder_location)
        
        if (len(output_names) == len(result)):
            for i in range(len(output_names)):
                name = output_names[i]
                data = result[i]
                if type(data)==dict:
                    if 'hdf5_dict' in data:
                        dataset = create_dataset_from_dict(f[out_folder_location], name, data)
                        if prop_attrs is not None:
                            dataset = propagate_attrs(dataset, prop_attr_keys, prop_attr_vals)
                    else:
                        dataset = f[out_folder_location].create_dataset(name, data=data)
                        if prop_attrs is not None:
                            dataset = propagate_attrs(dataset, prop_attr_keys, prop_attr_vals)
                else:
                    dataset = f[out_folder_location].create_dataset(name, data=data)
                    if prop_attrs is not None:
                        dataset = propagate_attrs(dataset, prop_attr_keys, prop_attr_vals)
                write_generic_attributes(fproc[name], out_folder_location + name, in_paths, name)
        else:
            print('Error: Unequal amount of outputs and output names')
        for key, value in kwargs.items():
            dataset.attrs[key] = value
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
# outputs_names (default: None): list of the names of the channels for the writting of the results.
#     By default, copies names from the first of the in_paths
# folder_names (default: None): list of the names of the folder containing results data channels.
#     By default, copies names from the first of the in_paths
# prop_attrs (default: None): string, or list of strings, that are the names of attributes that will
#     be copied from in_paths, into each output file. If the same attribute name is in multiple
#     in_paths, the first in_path with the attribute name will be copied from.
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

def l_apply(filename, function, all_input_criteria, output_names = None, folder_names = None, 
            prop_attrs = None, repeat = None, **kwargs):
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
        if len(set(list_lengths)) == 0:
            print('No Input Datafiles found!')
        elif len(set(list_lengths)) != 1:
            if repeat is None:
                print('Input lengths not equal, and repeat not set! Extra files will be omitted.')
            else:
                largest_list_length = np.max(list_lengths)
                list_multiples = []
                for length in list_lengths:
                    if largest_list_length%length != 0:
                        print('At least one path list length is not a factor of the largest path'\
                              'list length. Extra files will be omitted.')
                    list_multiples.append(largest_list_length//length)
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
        all_in_path_list = list(map(list, zip(*all_in_path_list)))
        
    increment_proc = True
    start_time = time.time()
    for path_num in range(len(all_in_path_list)):
        m_apply(filename, function, all_in_path_list[path_num], output_names = output_names,
                folder_names = folder_names, increment_proc = increment_proc,
                prop_attrs = prop_attrs, **kwargs)
        progress_report(path_num+1, len(all_in_path_list), start_time, function.__name__,
                        all_in_path_list[path_num])
        increment_proc = False    
    
        
#   FUNCTION create_dataset_from_dict
# Subfunction used in m_apply. Converts the hdf5_dict output file into a dataset that is written to
# the hdf5 file, with all attributes encoded.
#   INPUTS:
# dataset: Path to the datasets in the hdf5 file that contain the input.
# name: name of the channel the results are written in.
# dict_data: the dict that contains the data written. This should include both the actual data
#     to be written, as well as additional attributes.
#   OUTPUTS:
# dataset: The dataset written to

def create_dataset_from_dict (dataset, name, dict_data):
    dataset = dataset.create_dataset(name, data = dict_data['data'])
    for key, value in dict_data.items():
        if (key != 'hdf5_dict') and (key != 'data'):
            dataset.attrs[key] = value
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

def propagate_attrs (dataset, prop_attr_keys = [], prop_attr_vals = []):
    for i_attr in range(len(prop_attr_keys)):
        dataset.attrs[prop_attr_keys[i_attr]]=prop_attr_vals[i_attr]
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
        'hdf5_dict':True,
        'data':data
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

def write_generic_attributes(dataset, out_folder_location, in_paths, output_name):
    if type(in_paths) != list:
        in_paths = [in_paths]
    operation_name = out_folder_location.split('/')[1]
    dataset.attrs['shape'] = dataset.shape
    dataset.attrs['name'] = output_name
    dataset.attrs['operation name'] = operation_name.split('-')[1]
    dataset.attrs['operation number'] = operation_name.split('-')[0]
    dataset.attrs['time'] = str(datetime.now())
    for i in range(len(in_paths)):
        dataset.attrs['source' + str(i)] = in_paths[i]

        
#   FUNCTION progress_report
# Prints progression of a process run in several stages
#   INPUTS:
# processes_complete: number of processes that have currently been run
# processes_total: total number of processes that have been and will be run
# start_time: time at which the process began
# process_name: name of the process run
# identifier: name of the particular iteration of the process
# clear (default: True): Decides whether to overwrite existing print statements
#   OUTPUTS
# null

def progress_report(processes_complete, processes_total, start_time = None,
                    process_name = 'undefined_process', identifier = '[unidentified_sample]',
                    clear = True):
    if processes_complete != processes_total:
        if start_time is not None:
            time_remaining = round(((processes_total-processes_complete)/processes_complete)*
                                   (time.time()-start_time))
            str_progress = (process_name+': ' +str(processes_complete)+' of '+str(processes_total) +
                            ' Complete. ' + str(time_remaining)+'s remaining. '+str(identifier))
        else:
            str_progress = (process_name+': ' +str(processes_complete)+' of '+str(processes_total) +
                            ' Complete. Unknown time remaining. '+str(identifier))
        if clear:
            print(str_progress+' '*len(str_progress), sep=' ', end='\r', file=sys.stdout, flush=False)
        else:
            print(str_progress)
    if processes_complete == processes_total:
        str_final = (process_name+' complete! '+str(processes_complete)+' processes performed in '
                     + str(round(time.time()-start_time)) +'s')
        if clear:
            print(str_final+' '*100, sep=' ', end='\r', file=sys.stdout, flush=False)
        else:
            print(str_final)
        

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


           
#   FUNCTION find_paths_of_all_subgroups
# Recursively determines list of paths for all datafiles in current_path, as well as datafiles in
# all  subfolders (and sub-subfolders and...) of current path
#   INPUTS:
# f: open hdf5 file
# current_path: current group searched
#   OUTPUTS:
# path_list: list of paths to datafiles

def find_paths_of_all_subgroups(f, current_path):
    path_list = []
    for sub_group in f[current_path]:
        if isinstance(f[current_path + '/' + sub_group], h5py.Group):
            path_list.extend(find_paths_of_all_subgroups(f, current_path + '/' + sub_group))
        elif isinstance(f[current_path + '/' + sub_group], h5py.Dataset):
            path_list.append(current_path + '/' + sub_group)
    return path_list


# FUNCTION negative_
## Processes an array determine negatives of all values
## Trivial sample function to show how to use proc_tools
# INPUTS:
## filename: name of hdf5 file containing data
## data_folder: folder searched for inputs. eg. 'datasets', or 'process/negative'
## selection: determines the name of folders or files to be used. Can be None (selects all), a string, or a list of strings
## criteria: determines category of files to search
# OUTPUTS:
## null

def negative(filename, data_folder='datasets', selection=None, criteria=0):
    # Trivial sample function to show how to use proc_tools
    # Processes an array determine negatives of all values
    in_path_list = pt.path_inputs(filename, data_folder, selection, criteria)
    out_folder_locations = find_output_folder_location(filename, 'negative', in_path_list)
    with h5py.File(filename, "a") as f:
        for i in range(len(in_path_list)):
            neg = -np.array(f[in_path_list[i]])
            pt.write_output_f(f, neg, out_folder_locations[i], in_path_list[i])

####################################################################################################
####################################################################################################
# MOST USAGE OF THE BELOW FUNTIONS SHOULD BE REPLACED BY THE USE OF M_APPLY
# ONCE IT IS DONE WE SHOULD THINK IF WE NEED TO KEEP THEM OR NOT
####################################################################################################
####################################################################################################

#   FUNCTION propagate_scale_attrs
# Attempts to write the scale attributes to a new dataset. This is done by directly copying from
# an old dataset. If this is not possible, then it attempts to generate this from the old dataset
# by calculating from the 'size' and 'shape' attributes.
#   INPUTS:
# new_data: new dataset to write to
# old_data: old dataset to read from
#   OUTPUTS
# null

def propagate_scale_attrs(new_data, old_data):
    if 'scale (m/px)' in old_data.attrs:
        new_data.attrs['scale (m/px)'] = old_data.attrs['scale (m/px)']
    else:
        if ('size' in old_data.attrs) and ('shape' in old_data.attrs):
            scan_size = old_data.attrs['size']
            shape = old_data.attrs['shape']
            new_data.attrs['scale (m/px)'] = scan_size[0] / shape[0]


#   FUNCTION path_inputs
# Find paths to all input files
#   INPUTS:
# filename or f: either the open datafile, or the filename of an .hdf5 file that can be opened.
# data_folder (default: 'datasets'): folder searched for source data
# selection (default: None): determines the name of folders or files to be used.
# criteria: determines what category selection refers to. Can be either:
#     'process': an entire process folder
#     'sample': the next subfile after 'process' or 'dataset'; originally used for samples
#     'channel': the next subfile after 'sample'; originally the actual dataset, named after the
#         source channel.
#   OUTPUTS:
# in_path_list: list of paths to datafiles

def path_inputs(filename_or_f, data_folder='datasets', selection=None, criteria=None):
    if type(filename_or_f) == str:
        filename = filename_or_f
        with h5py.File(filename, 'a') as f:
            in_path_list = path_inputs(f, data_folder, selection, criteria)
    elif type(filename_or_f) == h5py._hl.files.File:
        f = filename_or_f
        in_path_list = find_paths_of_all_subgroups(f, data_folder)
        if selection is not None:
            in_path_list = criteria_selection(in_path_list, selection, criteria)
    else:
        print('Error: First argument should either be the filename of an hdf5 file, or an open\
                hdf5 file.')
        in_path_list = []
    return in_path_list

#   FUNCTION criteria_selection
# From a given list of paths, returns a shorter list that has been refined by particular criteria
#   INPUTS:
# path_list: a list of all possible paths
# selection: determines the name of folders or files to be used.
# criteria: determines what category selection refers to. Can be either:
#     'process': an entire process folder
#     'sample': the next subfile after 'process' or 'dataset'; originally used for samples
#     'channel': the next subfile after 'sample'; originally the actual dataset, named after the
#         source channel.
#   OUTPUTS:
# valid_path_list: shorter list of paths

def criteria_selection(path_list, selection, criteria):
    valid_path_list = []
    if type(selection) != list:
        selection = [selection]
    if criteria == 'process' or criteria == 'Process' or criteria == 'p':
        for path in path_list:
            split_path = path.split('/')
            if split_path[0] == 'process':
                if is_valid_path(split_path, selection, 1):
                    valid_path_list.append(path)
            else:
                print('Error: \'process\' criteria given, but path not linking to \'process\' '\
                        'folder. Path considered invalid.')

    elif criteria == 'sample' or criteria == 'Sample' or criteria == 's':
        for path in path_list:
            split_path = path.split('/')
            if split_path[0] == 'process':
                if is_valid_path(split_path, selection, 2):
                    valid_path_list.append(path)
            elif split_path[0] == 'datasets':
                if is_valid_path(split_path, selection, 1):
                    valid_path_list.append(path)
            else:
                print('Error: Path not linking to \'process\' or \'datasets\' folder. Path '\
                         'considered invalid.')
    elif criteria == 'channel' or criteria == 'Channel' or criteria == 'c':
        for path in path_list:
            split_path = path.split('/')
            if split_path[0] == 'process':
                if is_valid_path(split_path, selection, 3):
                    valid_path_list.append(path)
            elif split_path[0] == 'datasets':
                if is_valid_path(split_path, selection, 2):
                    valid_path_list.append(path)
            else:
                print('Error: Path not linking to \'process\' or \'datasets\' folder. Path '\
                         'considered invalid.')
    elif criteria is None:
        for path in path_list:
            split_path = path.split('/')
            if is_valid_path(split_path, selection, 0):
                valid_path_list.append(path)
    else:
        print('Error: Criteria should be either \'process\', \'sample\', or \'channel\'. No '\
                 'selection was performed.')
        valid_path_list = path_list
    return valid_path_list

#   FUNCTION is_valid_path
# Determines if the path is one of the selected, valid paths.
#   INPUTS:
# split_path: the path, split in a list along each iteration of the character '/'
# selection: string that is required for the path to be valid
# check_index: index of the split path checked to contain the selection
#   OUTPUTS:
# valid: bool that states if the path is valid or not

def is_valid_path(split_path, selection, check_index):
    valid = False
    for valid_name in selection:
        if split_path[check_index] == valid_name:
            valid = True
    return valid


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
        if overwrite_if_same == True:
            if str(operation_number - 1) + '-' + process_name in f['process'].keys():
                operation_number = operation_number - 1
        if type(folder_names) != list:
            folder_names = [folder_names]
        for folder in folder_names:
            if '/' in folder:
                folder_root, output_filename = folder.rsplit('/', 1)
                if folder_root.split('/', 1)[0] == 'datasets':
                    folder_centre = folder_root.split('/', 1)[1]
                elif folder_root.split('/', 1)[0] == 'process':
                    folder_centre = folder_root.split('/', 2)[2]
                else:
                    print('Error: folder_names should not contain a slash unless a path to either '\
                           'datasets or process')
            else:
                folder_centre = folder
            out_folder_location = ('process/' + str(operation_number).zfill(3) + '-' + process_name + '/'
                                   + folder_centre + '/')
            out_folder_location_list.append(out_folder_location)
    return out_folder_location_list

#   FUNCTION write_output_f
# Writes the output to an open datafile
#   INPUTS:
# f: the open datafile
# data: the data to be written
# out_folder_location: location the dataset is written to
# in_paths: list of paths for the sources of the data
# output_name (default: None): the name of the datafile to be written. If left as None, the output
#     name is inherited from the name of the first entry of in_paths
#   OUTPUTS
# dataset: the directory to the dataset, such that f[dataset] would yield data

def write_output_f(f, data, out_folder_location, in_paths, output_name=None):
    f.require_group(out_folder_location)
    if type(in_paths) != list:
        in_paths = [in_paths]
    if output_name is None:
        if type(in_paths[0]) == str:
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
    write_generic_attributes(dataset, out_folder_location, in_paths, output_name)
    return dataset


####################################################################################################
####################################################################################################
# DEPRACATED FILES
# DO NOT USE
####################################################################################################
####################################################################################################
# DELETE WHEN NO LONGER USED BY ANY FUNCTIONS


# FUNCTION initialise_process
## Initialises typical processes by finding paths to inputs, creating output folder, and finding their path
# INPUTS:
## filename: name of hdf5 file containing data
## process_name: name of the folder created
## create_groups: if True, creates folders, otherwise does not. Used for debugging.
## data_folder: folder searched for inputs. eg. 'datasets', or 'process/negative'
## selection: determines the name of folders or files to be used. Can be None (selects all), a string, or a list of strings
## selection_depth: determines what level at which to look at a selection.
## labelsize (default: 25) : Size of the text in pxs
## std_range (default: 3) : Range around the mean for the colorscale, alternatively the value can be "full", to take the full range.
## saving_path (default: '') : The path to the folder where to save the image
# OUTPUTS:
## Returns a list for each file input. Each entry of this list is a three entry list.
## This sublist consists of the path to the file input, the path to the file output, and the name of the file output
## If process_name is left as None, the output folder creation is bypassed, and a simple list of in_paths are provided

def initialise_process(filename, process_name=None, data_folder='datasets', selection=None, selection_depth=0,
                       create_groups=True):
    with h5py.File(filename, 'a') as f:
        operation_number = str(len(f['process']) + 1)
        # Find Inputs
        in_path_list = find_paths_of_all_subgroups(f, data_folder)
        if selection is not None:
            in_path_list = select_certain_data(in_path_list, selection, selection_depth)
        # print(f['process'])

        if process_name is not None:
            # Create Output Folder
            out_path_list = []
            output_filename_list = []
            for path in in_path_list:
                path_root, output_filename = path.rsplit('/', 1)
                output_filename_list.append(output_filename)
                out_folder = path_root[len(data_folder):]  # Skips initial nesting
                out_folder = 'process/' + operation_number + '-' + process_name + out_folder + '/'
                out_path_list.append(out_folder)
                if create_groups == True:
                    try:
                        curr_group = f.create_group(out_folder)
                    except:
                        pass

            in_out_list_of_lists = []
            for i in range(len(in_path_list)):
                in_out_list_of_lists.append([in_path_list[i], out_path_list[i], output_filename_list[i]])
        else:
            in_out_list_of_lists = in_path_list
    return in_out_list_of_lists


# FUNCTION select_certain_data
## Selects valid paths from a list of larger paths
# INPUTS:
## path_list: initial (longer) list of paths
## selection: determines the name of folders or files to be used. Can be None (selects all), a string, or a list of strings
## selection_depth: determines what level at which to look at a selection.
# OUTPUTS:
## shortened list of paths

def select_certain_data(path_list, selection, selection_depth):
    # Given a list of paths, returns a shorter list given the selection_params. See initialise_process
    if type(selection) != list:
        selection = [selection]
    valid_path_list = []
    for i in range(len(path_list)):
        split_path = path_list[i].split('/')
        valid = False
        for valid_name in selection:
            if split_path[selection_depth] == valid_name:
                valid = True
        if valid == True:
            valid_path_list.append(path_list[i])
    return valid_path_list


# FUNCTION generic_write
# Writes output for most "generic" datafiles, overwriting by default
# Logs source folder, written name, shape, operation name and number, and current time in attributes
# INPUTS:
## f: open hdf5 file
## data: data to be written into hdf5
## path: path list of lists; the output of initialise_process
# OUTPUTS:
## null

def generic_write(f, path, data=None, *arg):
    if data != None:
        try:
            # By default doesn't allow overwrite, so delete before writing
            del f[path[1]][path[2]]
        except:
            pass
        f[path[1]].create_dataset(path[2], data=data)
        f[path[1]][path[2]].attrs['shape'] = data.shape

    operation_name = path[1].split('/')[1]
    f[path[1]][path[2]].attrs['name'] = path[2]
    f[path[1]][path[2]].attrs['operation name'] = operation_name.split('-')[1]
    f[path[1]][path[2]].attrs['operation number'] = operation_name.split('-')[0]
    f[path[1]][path[2]].attrs['source'] = path[0]
    f[path[1]][path[2]].attrs['time'] = str(datetime.now())

    if data == None:
        f[path[1]][path[2]].attrs['shape'] = f[path[1]][path[2]][()].shape
    if len(arg) % 2 != 0:
        print('Error: Odd amount of input arguments')
    else:
        for i in range(len(arg) // 2):
            f[path[1]][path[2]].attrs[str(arg[2 * i])] = arg[2 * i + 1]


# FUNCTION find_data_path_structure
## Helper/debugger function.
## Prints paths to valid datafiles given a filename, a data_folder to look in, and selection params
## Intended to help user to determine selection params for initialise_process
# INPUTS:
## filename: name of hdf5 file containing data
## data_folder: folder searched for inputs. eg. 'datasets', or 'process/negative'
## selection: determines the name of folders or files to be used. Can be None (selects all), a string, or a list of strings
## selection_depth: determines what level at which to look at a selection.
# OUTPUTS:
## null

def find_data_path_structure(filename, data_folder='datasets', selection=None, selection_depth=0):
    with h5py.File(filename, 'a') as f:
        in_path_list = find_paths_of_all_subgroups(f, data_folder)
        if selection is not None:
            in_path_list = select_certain_data(in_path_list, selection, selection_depth)
    print(in_path_list)
