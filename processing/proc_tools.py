import h5py

def find_paths_of_all_subgroups(f, current_path):
    path_list = []
    for sub_group in f[current_path]:
        if isinstance(f[current_path+'/'+sub_group], h5py.Group):
            path_list.extend(find_paths_of_all_subgroups(f, current_path+'/'+sub_group))
        elif isinstance(f[current_path+'/'+sub_group], h5py.Dataset):
            path_list.append(current_path+'/'+sub_group)
    return path_list

def select_certain_data(path_list, selection, selection_depth):
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

def initialise_process(filename, process_name, create_groups = True, data_folder = 'datasets', selection = None, selection_depth = 0):
    with h5py.File(filename, 'a') as f:
        # Find Inputs
        in_path_list = find_paths_of_all_subgroups(f, data_folder)
        if selection is not None:
            in_path_list = select_certain_data(path_list, selection_depth, selection)

        # Create Output Folder
        out_path_list = []
        output_filename_list = []
        for path in in_path_list:
            path_root, output_filename = path.rsplit('/', 1)
            output_filename_list.append(output_filename)
            out_folder = path_root[len(data_folder):]      # Skips initial nesting
            out_folder = 'process/'+process_name+out_folder+'/'
            out_path_list.append(out_folder)
            if create_groups == True:
                try:
                    curr_group = f.create_group(out_folder)
                except:
                    pass
        
        in_out_list_of_lists = []
        for i in range(len(in_path_list)):
            in_out_list_of_lists.append([in_path_list[i], out_path_list[i], output_filename_list[i]])
    return in_out_list_of_lists

def generic_write(f, data, path):
    try:
        # By default doesn't allow overwrite, so delete before writing
        del f[path[1]][path[2]]
    except:
        pass
    f[path[1]].create_dataset(path[2], data = data)
    f[path[1]][path[2]].attrs['source'] = path[0]
    f[path[1]][path[2]].attrs['name'] = path[2]
    f[path[1]][path[2]].attrs['shape'] = data.shape
    