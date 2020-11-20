import h5py

import numpy as np
import cv2
import time
from . import core
from . import twodim
from skimage import img_as_ubyte

#   FUNCTION l_apply_classic
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
# use_attrs (default: None): string, or list of strings, that are the names of attributes that will
#     be copied from in_paths, and passed into the function as a kwarg for use.
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
#    TO DO:
# Can we force a None to be passed?


def l_apply_classic(filename, function, all_input_criteria, output_names=None, folder_names=None,
                    use_attrs=None, prop_attrs=None, repeat=None, **kwargs):
    print('This function is deprecated, please try to make use of l_apply instead.')
    all_in_path_list = core.path_search(filename, all_input_criteria, repeat)
    all_in_path_list = list(map(list, zip(*all_in_path_list)))
    increment_proc = True
    start_time = time.time()
    for path_num in range(len(all_in_path_list)):
        core.m_apply(filename, function, all_in_path_list[path_num], output_names=output_names,
                folder_names=folder_names, increment_proc=increment_proc,
                use_attrs=use_attrs, prop_attrs=prop_attrs, **kwargs)
        core.progress_report(path_num + 1, len(all_in_path_list), start_time, function.__name__,
                        all_in_path_list[path_num])
        increment_proc = False


def distortion_params_classic(filename, all_input_criteria, speed=2, read_offset=False,
                       cumulative=False, filterfunc=twodim.normalise):
    """
    Determine cumulative translation matrices for distortion correction and directly write it into
    an hdf5 file

    Parameters
    ----------
    filename : str
        name of hdf5 file containing data
    all_input_criteria : str
        criteria to identify paths to source files using pt.path_search. Should be
        height data to extract parameters from
    speed : int, optional
        Value between 1 and 4, which determines speed and accuracy of function. A higher number is
        faster, but assumes lower distortion and thus may be incorrect. Default value is 2.
    read_offset : bool, optional
        If set to True, attempts to read dataset for offset attributes to
        improve initial guess and thus overall accuracy (default is False).
    cumulative : bool, optional
        Determines if each image is compared to the previous image (default,
        False), or to the original image (True). Output format is identical.
    fitlerfunc : func, optional
        Function applied to image before identifying distortion params

    Returns
    -------
        None
    """

    print('This function is deprecated, please try to make use of distortion_params_ instead.')

    in_path_list = core.path_search(filename, all_input_criteria)[0]
    out_folder_locations = core.find_output_folder_location(filename, 'distortion_params',
                                                          in_path_list)
    tform21 = np.eye(2, 3, dtype=np.float32)
    cumulative_tform21 = np.eye(2, 3, dtype=np.float32)
    with h5py.File(filename, "a") as f:
        recent_offsets = []
        for i in range(len(in_path_list)):
            if i == 0:
                start_time = time.time()
            else:
                print('---')
                print('Currently reading path ' + in_path_list[i])
                i1 = f[in_path_list[0]]
                if (i > 1) and (not cumulative):
                    i1 = f[in_path_list[i-1]]
                i2 = f[in_path_list[i]]
                if filterfunc is not None:
                    i1 = filterfunc(i1)
                    i2 = filterfunc(i2)
                img1 = img_as_ubyte(i1)
                img2 = img_as_ubyte(i2)

                # try estimate offset change from attribs of img1 and img2
                if read_offset:
                    offset2 = (f[in_path_list[i]]).attrs['offset']
                    offset1 = (f[in_path_list[i - 1]]).attrs['offset']
                    scan_size = (f[in_path_list[i]]).attrs['size']
                    shape = (f[in_path_list[i]]).attrs['shape']
                    offset_px = twodim.m2px(offset2 - offset1, shape, scan_size)
                else:
                    offset_px = np.array([0, 0])
                if speed != 0 and speed != 1 and speed != 2 and speed != 3 and speed != 4:
                    print('Error: Speed should be an integer between 1 (slowest) and 4 (fastest).\
                            Speed now set to level 2.')
                    speed = 2
                if len(recent_offsets) == 0:
                    offset_guess = offset_px
                    if speed == 1:
                        warp_check_range = 16
                    elif speed == 2:
                        warp_check_range = 12
                    elif speed == 3:
                        warp_check_range = 10
                    elif speed == 4:
                        warp_check_range = 8
                elif len(recent_offsets) < 3:
                    offset_guess = offset_px + recent_offsets[-1]
                    if speed == 1:
                        warp_check_range = 12
                    elif speed == 2:
                        warp_check_range = 8
                    elif speed == 3:
                        warp_check_range = 8
                    elif speed == 4:
                        warp_check_range = 6
                else:
                    offset_guess = (offset_px + recent_offsets[2] / 2 + recent_offsets[1] / 3
                                    + recent_offsets[0] / 6)
                    # if i == 9:
                    #    offset_guess = offset_guess-np.array([20,20])
                    #    print(offset_guess)
                    if speed == 1:
                        warp_check_range = 8
                    elif speed == 2:
                        warp_check_range = 6
                    elif speed == 3:
                        warp_check_range = 4
                    elif speed == 4:
                        warp_check_range = 2
                if (offset_px[0] != 0) or (offset_px[1] != 0):
                    print('Offset found from file attributes: ' + str(offset_px))
                    warp_check_range = warp_check_range + 8
                    recent_offsets = []
                tform21 = generate_transform_xy_classic(img1, img2, tform21, offset_guess,
                                                warp_check_range, cumulative, cumulative_tform21)
                if cumulative:
                    tform21[0, 2] = tform21[0, 2] - cumulative_tform21[0, 2]
                    tform21[1, 2] = tform21[1, 2] - cumulative_tform21[1, 2]
                cumulative_tform21[0, 2] = cumulative_tform21[0, 2] + tform21[0, 2]
                cumulative_tform21[1, 2] = cumulative_tform21[1, 2] + tform21[1, 2]
                print('Scan ' + str(i) + ' Complete. Cumulative Transform Matrix:')
                print(cumulative_tform21)
                if (offset_px[0] == 0) and (offset_px[1] == 0):
                    recent_offsets.append([tform21[0, 2], tform21[1, 2]] - offset_px)
                    if len(recent_offsets) > 3:
                        recent_offsets = recent_offsets[1:]
            data = core.write_output_f(f, cumulative_tform21, out_folder_locations[i],
                                     in_path_list[i])
            core.progress_report(i + 1, len(in_path_list), start_time, 'distortion_params',
                               in_path_list[i], clear=False)


def distortion_correction_classic(filename, all_input_criteria, cropping=True):
    """
    Applies distortion correction parameters to an image. The distortion corrected data is then
    cropped to show only the common data, or expanded to show the maximum extent of all possible data.

    Parameters
    ----------
    filename : str
        Filename of hdf5 file containing data
    all_input_criteria : list
        Criteria to identify paths to source files using pt.path_search. First should
        be data to be corrected, second should be the distortion parameters.
    cropping : bool, optional
        If set to True, each dataset is cropped to show only the common area. If
        set to false, expands data shape to show all data points of all images. (default: True)

    Returns
    -------
    None
    """
    print('This function is deprecated, please try to make use of distortion_correction_ instead.')

    all_in_path_list = core.path_search(filename, all_input_criteria, repeat='block')
    in_path_list = all_in_path_list[0]
    dm_path_list = all_in_path_list[1]

    distortion_matrices = []
    with h5py.File(filename, "a") as f:
        for path in dm_path_list[:]:
            distortion_matrices.append(np.copy(f[path]))
        xoffsets = []
        yoffsets = []
        for matrix in distortion_matrices:
            xoffsets.append(np.array(matrix[0, 2]))
            yoffsets.append(np.array(matrix[1, 2]))
    offset_caps = [np.max(xoffsets), np.min(xoffsets), np.max(yoffsets), np.min(yoffsets)]

    out_folder_locations = core.find_output_folder_location(filename, 'distortion_correction',
                                                          in_path_list)

    with h5py.File(filename, "a") as f:
        start_time = time.time()
        for i in range(len(in_path_list)):
            orig_image = f[in_path_list[i]]
            if cropping:
                final_image = twodim.array_cropped(orig_image, xoffsets[i], yoffsets[i], offset_caps)
            else:
                final_image = twodim.array_expanded(orig_image, xoffsets[i], yoffsets[i], offset_caps)
            data = core.write_output_f(f, final_image, out_folder_locations[i], [in_path_list[i],
                                                                               dm_path_list[i]])
            twodim.propagate_scale_attrs(data, f[in_path_list[i]])
            core.progress_report(i + 1, len(in_path_list), start_time, 'distortion_correction',
                               in_path_list[i])


def generate_transform_xy_classic(img, img_orig, tfinit=None, offset_guess = [0,0], warp_check_range=10,
                          cumulative=False, cumulative_tform21=np.eye(2,3,dtype=np.float32)):
    """
    Determines transformation matrices in x and y coordinates

    Parameters
    ----------
    img : cv2
        Currently used image (in cv2 format) to find transformation array of
    img_orig : cv2
        Image (in cv2 format) transformation array is based off of
    tfinit : array_like or None, optional
        Base array passed into function
    offset_guess : list, optional
        Array showing initial estimate of distortion, in pixels (default: [0,0])
    warp_check_range : int, optional
        Distance (in pixels) that the function will search to find the optimal transform matrix.
        Number of iterations = (warp_check_range+1)**2. (default: 10)
    cumulative : bool, optional
        Determines if each image is compared to the previous image (default, False), or to the original image (True).
        Output format is identical.
    cumulative_tform21 : ndarray, optional
        The transformation matrix, only used if cumulative is switched to True. (default: np.eye(2,3,dtype=np.float32))

    Returns
    -------
    warp_matrix : ndarray
        Transformation matrix used to convert img_orig into img
    """

    print('This function is deprecated, please try to make use of gernerate_transform_xy_ instead.')

    # Here we generate a MOTION_EUCLIDEAN matrix by doing a
    # findTransformECC (OpenCV 3.0+ only).
    # Returns the transform matrix of the img with respect to img_orig
    warp_mode = cv2.MOTION_TRANSLATION
    if tfinit is not None:
        warp_matrix = tfinit
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    number_of_iterations = 10000
    termination_eps = 1e-3
    term_flags = cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT

    if cumulative:
        offset_guess[0] = offset_guess[0] + cumulative_tform21[0, 2]
        offset_guess[1] = offset_guess[1] + cumulative_tform21[1, 2]

    criteria = (term_flags, number_of_iterations, termination_eps)

    diff = np.Inf
    offset1 = 0
    offset2 = 0
    for i in range(-warp_check_range // 2, (warp_check_range // 2) + 1):
        for j in range(-warp_check_range // 2, (warp_check_range // 2) + 1):
            warp_matrix[0, 2] = 2 * i + offset_guess[0]
            warp_matrix[1, 2] = 2 * j + offset_guess[1]
            try:
                (cc, tform21) = cv2.findTransformECC(img_orig, img, warp_matrix, warp_mode,
                                                     criteria)
                img_test = cv2.warpAffine(img, tform21, (512, 512), flags=cv2.INTER_LINEAR +
                                                                          cv2.WARP_INVERSE_MAP)
                currDiff = np.sum(np.square(img_test[150:-150, 150:-150]
                                            - img_orig[150:-150, 150:-150]))
                if currDiff < diff:
                    diff = currDiff
                    offset1 = tform21[0, 2]
                    offset2 = tform21[1, 2]
            except:
                pass
            warp_matrix[0, 2] = offset1
            warp_matrix[1, 2] = offset2
    return warp_matrix