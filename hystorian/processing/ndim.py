from . import core
import numpy as np
import h5py
import matplotlib.pyplot as plt


# FUNCTION extract_hist
## Split the SSPFM curve into on and off signal.
## Function was rewritten to work with m_apply, and to take account a bug into Cypher machine, leading to pulse not
## all being the same size.
# INPUTS:
## input_data: data which needs to be splitted
## data_folder: data of the bias waveform applied during SSPFM
# OUTPUTS:
## return two grid with the data splitted into "on/off" according to the bias shape

def extract_hist(*input_data, bias):
    idx_zeroes = np.where(bias[0, 0, :] == 0)[0]
    idx_Nonzeroes = np.where(bias[0, 0, :] != 0)[0]

    v1 = idx_zeroes[:-1]
    v2 = np.roll(idx_zeroes, 1)[:-1]
    list_change_zeroes = list(np.where((v1 - v2) != 1)[0])

    idx_start = idx_zeroes[list_change_zeroes]

    v1 = idx_Nonzeroes[:-1]
    v2 = np.roll(idx_Nonzeroes, 1)[:-1]
    list_change_nonzeroes = list(np.where((v1 - v2) != 1)[0])
    idx_end = idx_Nonzeroes[list_change_nonzeroes]

    data_k_on = np.ndarray((np.shape(bias)[0], np.shape(bias)[1], len(idx_end) - 1))
    data_k_off = np.ndarray((np.shape(bias)[0], np.shape(bias)[1], len(idx_end) - 1))

    input_data = input_data[0]
    for i in range(len(idx_end) - 1):
        data_k_on[:, :, i] = np.median(input_data[:, :, idx_end[i]:idx_start[i + 1]], axis=2)
        data_k_off[:, :, i] = np.median(input_data[:, :, idx_start[i + 1]:idx_end[i + 1]], axis=2)

    output = [data_k_on, data_k_off]
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


def gauss_area(x, y):
    """
    Determine the area created by the polygon formed by x,y using the Gauss's area formula (also called shoelace formula)

    Parameters
    ----------
    x  : array_like
        values along the x axis
    y : array_like
        values along the x axis

    Returns
    -------
        float
            value corresponding at the encompassed area
    """

    area = 0.0
    for i in range(len(x)):
        x1 = x[i]
        y1 = y[i]

        if i < len(x) - 1:
            x2 = x[i + 1]
            y2 = y[i + 1]
        else:
            x2 = x[0]
            y2 = y[0]

        area = area + x1 * y2 - x2 * y1
    return np.abs(area / 2.0)


def clean_loop(bias, phase, amp, threshold=None, debug=False):
    '''
    Used to determine if a SSPFM loop is good or not by calculating the area encompassed by the hysteresis curve and
    comparing it to a threshold

    Parameters
    ----------
    bias: nd-array
        3d-array containing the bias applied to each point of the grid
    phase: nd-array
        3d-array containing the phase applied to each point of the grid
    amp: nd-array
        3d-array containing the amplitude applied to each point of the grid
    threshold: int or float, optional
        minimal value of the loop area to be considered a good loop (default: None)
        if set to None will use threshold = np.mean(area_grid_full) - 2 * np.std(area_grid_full)
    debug: bool, optional
        if set to True will use plt.imshow() to plot the loops in red for bad and blue for good. (default: False)

    Returns
    -------
        list_bias: array-like
            list of the bias corresponding to good loops
        list_phase: array-like
            list of the phase corresponding to good loops
        list_amp: array-like
            list of the amplitudes corresponding to good loops
        mask: nd-array
            a 2D mask where a 1 correspond to a good loop and 0 to a bad loop, can be used to mask the input data.
    '''
    list_bias = []
    list_phase = []
    list_amp = []
    list_bias_bad = []
    list_phase_bad = []
    mask = np.ndarray((np.shape(bias)[0], np.shape(bias)[1]))

    if threshold is None:
        area_grid_full = np.ndarray((np.shape(bias)[0], np.shape(bias)[1]))
        for xi in range(np.shape(bias)[0]):
            for yi in range(np.shape(bias)[1]):
                area_grid_full[xi, yi] = gauss_area(bias[xi, yi, :], phase[xi, yi, :])
            threshold = np.mean(area_grid_full) - 2 * np.std(area_grid_full)

    for xi in range(np.shape(bias)[0]):
        for yi in range(np.shape(bias)[1]):
            if gauss_area(bias[xi, yi, :], phase[xi, yi, :]) > threshold:
                list_bias.append(bias[xi, yi, :])
                list_phase.append(phase[xi, yi, :])
                list_amp.append(amp[xi, yi, :])
                mask[xi, yi] = True
            else:
                mask[xi, yi] = False
                if debug:
                    list_bias_bad.append(bias[xi, yi, :])
                    list_phase_bad.append(phase[xi, yi, :])

    if debug:
        plt.figure(figsize=(10, 5))

        if threshold is None:
            plt.plot(np.ravel(area_grid_full))

        else:
            area_grid_full = np.ndarray((np.shape(bias)[0], np.shape(bias)[1]))
            for xi in range(np.shape(bias)[0]):
                for yi in range(np.shape(bias)[1]):
                    area_grid_full[xi, yi] = gauss_area(bias[xi, yi, :], phase[xi, yi, :])
            plt.plot(np.ravel(area_grid_full))

        plt.axhline(threshold, c='r')

    return list_bias, list_phase, list_amp, mask


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
