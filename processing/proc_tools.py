import h5py
from datetime import datetime

#FUNCTION initialise_process
## Initialises typical processes by finding paths to inputs, creating output folder, and finding their path
#INPUTS:
## filename: name of hdf5 file containing data
## process_name: name of the folder created
## create_groups: if True, creates folders, otherwise does not. Used for debugging.
## data_folder: folder searched for inputs. eg. 'datasets', or 'process/negative'
## selection: determines the name of folders or files to be used. Can be None (selects all), a string, or a list of strings
## selection_depth: determines what level at which to look at a selection.
## labelsize (default: 25) : Size of the text in pxs
## std_range (default: 3) : Range around the mean for the colorscale, alternatively the value can be "full", to take the full range.
## saving_path (default: '') : The path to the folder where to save the image
#OUTPUTS:
## Returns a list for each file input. Each entry of this list is a three entry list.
## This sublist consists of the path to the file input, the path to the file output, and the name of the file output
## If process_name is left as None, the output folder creation is bypassed, and a simple list of in_paths are provided

def initialise_process(filename, process_name = None, data_folder = 'datasets', selection = None, selection_depth = 0, create_groups = True):
    with h5py.File(filename, 'a') as f:
        operation_number = str(len(f['process'])+1)
        # Find Inputs
        in_path_list = find_paths_of_all_subgroups(f, data_folder)
        if selection is not None:
            in_path_list = select_certain_data(in_path_list, selection, selection_depth)
        #print(f['process'])

        if process_name is not None:
            # Create Output Folder
            out_path_list = []
            output_filename_list = []
            for path in in_path_list:
                path_root, output_filename = path.rsplit('/', 1)
                output_filename_list.append(output_filename)
                out_folder = path_root[len(data_folder):]      # Skips initial nesting
                out_folder = 'process/'+operation_number+'-'+process_name+out_folder+'/'
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


#FUNCTION find_paths_of_all_subgroups
## Recursively determines list of paths for all datafiles in current_path
## also finds datafiles in all subfolders (and sub-subfolders and...) of current path
#INPUTS:
## f: open hdf5 file
## current_path: current group searched
#OUTPUTS:
## list of paths to datafiles

def find_paths_of_all_subgroups(f, current_path):
    # Recursively determines list of paths for all datafiles in current_path, and all subfolders (and sub-subfolders and...) of current path
    path_list = []
    for sub_group in f[current_path]:
        if isinstance(f[current_path+'/'+sub_group], h5py.Group):
            path_list.extend(find_paths_of_all_subgroups(f, current_path+'/'+sub_group))
        elif isinstance(f[current_path+'/'+sub_group], h5py.Dataset):
            path_list.append(current_path+'/'+sub_group)
    return path_list


#FUNCTION select_certain_data
## Selects valid paths from a list of larger paths
#INPUTS:
## path_list: initial (longer) list of paths
## selection: determines the name of folders or files to be used. Can be None (selects all), a string, or a list of strings
## selection_depth: determines what level at which to look at a selection.
#OUTPUTS:
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


#FUNCTION generic_write
# Writes output for most "generic" datafiles, overwriting by default
# Logs source folder, written name, shape, operation name and number, and current time in attributes
#INPUTS:
## f: open hdf5 file
## data: data to be written into hdf5
## path: path list of lists; the output of initialise_process
#OUTPUTS:
## null

def generic_write(f, data, path, *arg):
    try:
        # By default doesn't allow overwrite, so delete before writing
        del f[path[1]][path[2]]
    except:
        pass
    operation_name = path[1].split('/')[1]
    f[path[1]].create_dataset(path[2], data = data)
    f[path[1]][path[2]].attrs['name'] = path[2]
    f[path[1]][path[2]].attrs['operation name'] = operation_name.split('-')[1]
    f[path[1]][path[2]].attrs['operation number'] = operation_name.split('-')[0]
    f[path[1]][path[2]].attrs['source'] = path[0]
    f[path[1]][path[2]].attrs['time'] = str(datetime.now())
    f[path[1]][path[2]].attrs['shape'] = data.shape
    if len(arg)%2 != 0:
        print('Error: Odd amount of input arguments')
    else:
        for i in range(len(arg)//2):
            f[path[1]][path[2]].attrs[str(arg[2*i])] = arg[2*i+1]

    
#FUNCTION find_data_path_structure
## Helper/debugger function.
## Prints paths to valid datafiles given a filename, a data_folder to look in, and selection params
## Intended to help user to determine selection params for initialise_process
#INPUTS:
## filename: name of hdf5 file containing data
## data_folder: folder searched for inputs. eg. 'datasets', or 'process/negative'
## selection: determines the name of folders or files to be used. Can be None (selects all), a string, or a list of strings
## selection_depth: determines what level at which to look at a selection.
#OUTPUTS:
## null
    
def find_data_path_structure(filename, data_folder = 'datasets', selection = None, selection_depth = 0):
    with h5py.File(filename, 'a') as f:
        in_path_list = find_paths_of_all_subgroups(f, data_folder)
        if selection is not None:
            in_path_list = select_certain_data(in_path_list, selection, selection_depth)
    print(in_path_list)
    

