from . import core
import numpy as np
import h5py

def extract_hist_(filename, bias_chan, channels_name, phase=1):
    with h5py.File(filename, 'a') as f:
        notedict = {}
        for file in f['metadata'].keys():
            for k in f['metadata'][file]:
                tmp = k.decode('utf-8').split(":", 1)
                try:
                    notedict[tmp[0]] = tmp[1]
                except:
                    pass

        #Extracting bias_length
        len_bias = len(f[bias_chan][1, 1, :])

    waveform_pulsetime = float(notedict["ARDoIVArg3"])
    if 'ARDoIVArg4' in notedict.keys():
        waveform_dutycycle = float(notedict["ARDoIVArg4"])
    else:
        waveform_dutycycle = 0.5

    waveform_delta = 1 / float(notedict["NumPtsPerSec"])
    waveform_numbiaspoints = int(np.floor(waveform_delta * len_bias / waveform_pulsetime))
    waveform_pulsepoints = int(waveform_pulsetime / waveform_delta)
    waveform_offpoints = int(waveform_pulsepoints * (1.0 - waveform_dutycycle))

    list_output_name = []
    for i in channels_name:
        list_output_name.append(i.split('/')[-1] + '_on')
        list_output_name.append(i.split('/')[-1] + '_off')

    core.m_apply(filename,
                 f_extract_hist,
                 in_paths=channels_name,
                 output_names=list_output_name,
                 len_bias=len_bias,
                 waveform_pulsetime=waveform_pulsetime,
                 waveform_dutycycle=waveform_dutycycle,
                 waveform_delta=waveform_delta)

    extract_list = []
    names_list = []
    with h5py.File(filename, 'a') as f:
        keys = f['process'].keys()
        for k in keys:
            if 'f_extract_hist' in k:
                extract_list.append(k)
                name_list = list(f['process/' + k].keys())

    for i in extract_list:
        for name in name_list:

            input_path = ['process/' + i + '/' + name + '/Bias_on']
            if phase == 1:
                input_path.append('process/' + i + '/' + name + '/Phase_off')
            elif phase == 2:
                input_path.append('process/' + i + '/' + name + '/Phas2_off')
            else:
                print('Wrong input for the phase choice, taking the first phase')
                input_path.append('process/' + i + '/' + name + '/Phase_off')

            core.m_apply(filename,
                         PFM_params_map,
                         in_paths=input_path,
                         output_names=['coerc_pos',
                                 'coerc_neg',
                                 'step_left',
                                 'step_right',
                                 'imprint',
                                 'phase_shift'])

def f_extract_hist(*chans, len_bias, waveform_pulsetime, waveform_dutycycle, waveform_delta):
    output = []
    waveform_numbiaspoints = int(np.floor(waveform_delta *len_bias / waveform_pulsetime))
    waveform_pulsepoints = int(waveform_pulsetime / waveform_delta)
    waveform_offpoints = int(waveform_pulsepoints * (1.0 - waveform_dutycycle))
    for chan in chans:
        result_on = np.ndarray(shape=(np.shape(chan)[0],np.shape(chan)[1], waveform_numbiaspoints))
        result_off= np.ndarray(shape=(np.shape(chan)[0],np.shape(chan)[1], waveform_numbiaspoints))
        for b in range(waveform_numbiaspoints):
            start = b * waveform_pulsepoints + waveform_offpoints
            stop = (b + 1) * waveform_pulsepoints

            var2 = stop - start + 1
            realstart = int(start + var2 * .25)
            realstop = int(stop - var2 * .25)
            result_on[:,:, b] = np.nanmean(chan[:, :, realstart:realstop], axis=2)
            start = stop
            stop = stop + waveform_pulsepoints * waveform_dutycycle

            var2 = stop - start + 1
            realstart = int(start + var2 * .25)
            realstop = int(stop - var2 * .25)
            result_off[:, :, b] = np.nanmean(chan[:, :, realstart:realstop], axis=2)
        output.append(result_on)
        output.append(result_off)

    output = tuple(output)
    return output


def calc_hyst_params(bias, phase):
    biasdiff = np.diff(bias)
    up = np.sort(np.unique(np.hstack((np.where(biasdiff > 0)[0], np.where(biasdiff > 0)[0] + 1))))
    dn = np.sort(np.unique(np.hstack((np.where(biasdiff < 0)[0], np.where(biasdiff < 0)[0] + 1))))

    # UP leg calculations
    x = np.array(bias[up])
    y = np.array((phase[up] + 360) % 360)
    step_left_up = np.median(y[np.where(x == np.min(x))[0]])
    step_right_up = np.median(y[np.where(x == np.max(x))[0]])

    avg_x = []
    avg_y = []
    for v in np.unique(x):
        avg_x.append(v)
        avg_y.append(np.mean(y[np.where(x == v)[0]]))

    my_x = np.array(avg_x)[1:] + 0 * (avg_x[0] + avg_x[1]) / 2.0
    my_y = np.abs(np.diff(avg_y))

    coercive_volt_up = my_x[np.nanargmax(my_y)]

    # DOWN leg calculations
    x = np.array(bias[dn])
    y = np.array((phase[dn] + 360) % 360)
    step_left_dn = np.median(y[np.where(x == np.min(x))[0]])
    step_right_dn = np.median(y[np.where(x == np.max(x))[0]])

    avg_x = []
    avg_y = []
    for v in np.unique(x):
        avg_x.append(v)
        avg_y.append(np.mean(y[np.where(x == v)[0]]))

    my_x = np.array(avg_x)[1:] + 0 * (avg_x[0] + avg_x[1]) / 2.0
    my_y = np.abs(np.diff(avg_y))

    coercive_volt_dn = my_x[np.nanargmax(my_y)]

    return [coercive_volt_up, coercive_volt_dn, 0.5 * step_left_dn + 0.5 * step_left_up,
            0.5 * step_right_dn + 0.5 * step_right_up]


def PFM_params_map(bias, phase):
    x, y, z = np.shape(phase)
    coerc_pos = np.zeros((x, y), dtype=float)
    coerc_neg = np.zeros((x, y), dtype=float)
    step_left = np.zeros((x, y), dtype=float)
    step_right = np.zeros((x, y), dtype=float)
    imprint = np.zeros((x, y), dtype=float)
    phase_shift = np.zeros((x, y), dtype=float)
    list_values = ['coerc_pos', 'coerc_neg', 'step_left', 'step_right', 'imprint', 'phase_shift']
    for xi in range(x):
        for yi in range(y):
            hyst_matrix = calc_hyst_params(bias[xi, yi, :], phase[xi, yi, :])
            coerc_pos[xi, yi] = hyst_matrix[0]
            coerc_neg[xi, yi] = hyst_matrix[1]
            step_left[xi, yi] = hyst_matrix[2]
            step_right[xi, yi] = hyst_matrix[3]
            imprint[xi, yi] = (hyst_matrix[0] + hyst_matrix[1]) / 2.0
            phase_shift[xi, yi] = (hyst_matrix[3] - hyst_matrix[2])

    return coerc_pos, coerc_neg, step_left, step_right, imprint, phase_shift


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

def negative_(filename, data_folder='datasets', selection=None, criteria=0):
    # Trivial sample function to show how to use proc_tools
    # Processes an array determine negatives of all values
    in_path_list = core.path_inputs(filename, data_folder, selection, criteria)
    out_folder_locations = core.find_output_folder_location(filename, 'negative', in_path_list)
    with h5py.File(filename, "a") as f:
        for i in range(len(in_path_list)):
            neg = -np.array(f[in_path_list[i]])
            core.write_output_f(f, neg, out_folder_locations[i], in_path_list[i])

# def negative_(filename, data_folder='datasets', selection = None, selection_depth = 0):
#    # Trivial sample function to show how to use proc_tools
#    # Processes an array determine negatives of all values
#    path_lists = pt.initialise_process(filename, 'negative', data_folder = data_folder, selection = selection, #selection_depth = selection_depth)
#    with h5py.File(filename, "a") as f:
#        for path in path_lists[:]:
#            neg = -np.array(f[path[0]])
#            pt.generic_write(f, path, neg)
