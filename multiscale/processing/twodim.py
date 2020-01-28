#234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
from . import core as pt
import cv2
import os
import time

from scipy.signal import medfilt, cspline2d
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, binary_dilation
from scipy.ndimage.measurements import label
from scipy import interpolate
from skimage.morphology import skeletonize
from random import randrange


#   FUNCTION save_image
# Saves one .png image to the current directory, or a chosen folder
#   INPUTS:
# data: A 2-D array which will be converted into a png image.
# image_name (default: None): name of the image that is saved. By default, tries to pull name from
#     source_path. If this cannot be done, sets name to 'image'
# colorm (default: 'inferno'): colormap to be used for image
# scalebar (default: False): if True, add a scalebar to the image, requires three attributes : 
#     shape, which define the pixel size of the image
#     size, which gives the phyiscal dimension of the image
#     unit, which give the physical unit of size
# physical_size (default: (0, 'unit')): physical size of the image used when generating the scalebar
# size (default: None): Dimension of the saved image. If none, the image is set to have one pixel 
#     per data point at 100 dpi
# labelsize (default: 16): Size of the text in pxs
# std_range (default: 3): Range around the mean for the colorscale, alternatively the value can be 
#    "full", to take the full range.
# saving_path (default: ''): The path to the folder where to save the image
# verbose (default: False): if True, print a line each time a image is saved.
# show (default: False): if True, the image is displayed in the kernel.
# source_path (default: None): if set, and image_name not set, this variable will be used to
#     generate the file name
#   OUTPUTS:
# null


def save_image(data,
               image_name=None, 
               colorm='inferno',
               scalebar=False,
               physical_size = (0, 'unit'),
               colorbar = True, 
               size=None, 
               labelsize=16, 
               std_range=3, 
               saving_path='', 
               verbose=False,
               show=False,
               source_path=None,
               source_scale_m_per_px=None):
    if data.dtype == 'bool':
         data = data.astype(int)
    if image_name is None:
        if source_path is not None:
            image_name = source_path.replace('/','_')
        else:
            image_name = 'image'
    if saving_path != '':
        if saving_path[-1] != '/':
            saving_path = saving_path +'/'
    if std_range != 'full':
        std_range = float(std_range)
    
        if size is None:
            figsize = (np.array(np.shape(data))/100)[::-1]
            if figsize[0] < 5:
                scale_factor = np.ceil(5/figsize[0])
                figsize = scale_factor*figsize
            fig = plt.figure(frameon=False, figsize=figsize, dpi=100)
        else:
            fig = plt.figure(figsize=size)

        plt.tick_params(labelsize=labelsize)
        offsetdata = data - np.nanmin(data)
        mean_val = np.nanmean(offsetdata)
        std_val = np.nanstd(offsetdata)
        v_min = 0
        v_max = mean_val + std_range*std_val
        if std_range == 'full':
            pos = plt.imshow(offsetdata, cmap=colorm)
        else:
            pos = plt.imshow(offsetdata, vmin=v_min, vmax=v_max, cmap=colorm)
        if colorbar == True:
            cbar = plt.colorbar(pos,fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=labelsize) 
        plt.tight_layout()
        if scalebar:
            try:
                if source_scale_m_per_px is None:
                    phys_size = physical_size[0]
                    px = np.shape(data)[0]
                    scalebar = ScaleBar(phys_size/px, physical_size[1], location='lower right',
                                    font_properties={'size':labelsize})
                else:
                    scalebar = ScaleBar(source_scale_m_per_px, 'm', location='lower right',
                                    font_properties={'size':labelsize})
                #scalebar = ScaleBar(f[path].attrs['scale (m/px)'], 'm', location='lower right',
                                    #font_properties={'size':25})
                plt.gca().add_artist(scalebar)
            except:
                print("Error in the creation of the scalebar, please check that the attribute's\
                        size and shape are correctly define for each data channel.")
        fig.savefig(saving_path+str(image_name)+'.png')
        if show:
            plt.show()
        if verbose:
            print(filename.split('.')[0]+'_'+str(image_name)+'.png saved.')
        plt.close()
    return


#   FUNCTION distortion_params_
# determine cumulative translation matrices for distortion correction.
#   INPUTS:
# filename: name of hdf5 file containing data
# data_folder (default: 'datasets'): folder searched for image inputs
# selection (default: None): determines the name of folders or files to be used.
# criteria (default: None): determines what category selection refers to
# speed (default: 2): int between 1 and 4, which determines speed and accuracy of function. A higher
#     number is faster, but assumes lower distortion and thus may be incorrect.
# read_offset (default: False): if set to True, attempts to read dataset for offset attributes to
#     improve initial guess and thus overall accuracy
# cumulative (default: False): determines if each image is compared to the previous image (default,
#      False), or to the original image (True). Output format is identical.
#   OUTPUTS
# null

def distortion_params_(filename, all_input_criteria, speed = 2, read_offset = False, cumulative = False):
    in_path_list = pt.path_search(filename, all_input_criteria)[0]
    out_folder_locations = pt.find_output_folder_location(filename, 'distortion_params',
                                                          in_path_list)
    tform21 = np.eye(2,3,dtype=np.float32)
    cumulative_tform21 = np.eye(2,3,dtype=np.float32)
    with h5py.File(filename, "a") as f:
        recent_offsets=[]
        for i in range(len(in_path_list)):
            if i == 0:
                start_time = time.time()
            else:
                print('---')
                print('Currently reading path '+in_path_list[i])
                if cumulative:
                    img1 = img2cv((f[in_path_list[0]]))
                    img2 = img2cv((f[in_path_list[i]]))
                else:
                    img1 = img2cv((f[in_path_list[i-1]]))
                    img2 = img2cv((f[in_path_list[i]]))
                
                # try estimate offset change from attribs of img1 and img2
                if read_offset == True:
                    offset2 = (f[in_path_list[i]]).attrs['offset']
                    offset1 = (f[in_path_list[i-1]]).attrs['offset']
                    scan_size = (f[in_path_list[i]]).attrs['size']
                    shape = (f[in_path_list[i]]).attrs['shape']
                    offset_px = m2px(offset2-offset1, shape, scan_size)
                else:
                    offset_px = np.array([0,0])
                if speed != 1 and speed != 2 and speed != 3 and speed != 4:
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
                    offset_guess = (offset_px + recent_offsets[2]/2 + recent_offsets[1]/3
                                    + recent_offsets[0]/6)
                    #if i == 9:
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
                tform21 = generate_transform_xy(img1, img2, tform21, offset_guess,
                                                warp_check_range, cumulative, cumulative_tform21)
                if cumulative:
                    tform21[0,2] = tform21[0,2]-cumulative_tform21[0,2]
                    tform21[1,2] = tform21[1,2]-cumulative_tform21[1,2]
                cumulative_tform21[0,2]=cumulative_tform21[0,2]+tform21[0,2]
                cumulative_tform21[1,2]=cumulative_tform21[1,2]+tform21[1,2]
                print('Scan '+str(i)+' Complete. Cumulative Transform Matrix:')
                print(cumulative_tform21)
                if (offset_px[0] == 0) and (offset_px[1] == 0):
                    recent_offsets.append([tform21[0,2], tform21[1,2]]-offset_px)
                    if len(recent_offsets)>3:
                        recent_offsets = recent_offsets[1:]
            data = pt.write_output_f(f, cumulative_tform21, out_folder_locations[i],
                                     in_path_list[i])
            pt.progress_report(i+1, len(in_path_list), start_time, 'distortion_params',
                            in_path_list[i], clear = False)


#   FUNCTION m2px
# Converts length in metres to a length in pixels
#   INPUTS:
# m: length in metres to be converted
# points: number of lines or points per row
# scan_size: total length of scan
#   OUTPUTS:
# px: converted length in pixels
        
def m2px (m, points, scan_size):
    px = m*points/scan_size
    return px


#   FUNCTION img2cv
# Converts image (numpy array, or hdf5 dataset) into cv2
#   INPUTS:
# img1: currently used image
# sigma_cutoff (default: 10): variation used to convert to cv-viable format
#   OUTPUTS
# img1: converted image into cv2 valid format

def img2cv(img1, sigma_cutoff=10):
    img1 = img1-np.min(img1) 
    img1 = img1/np.max(img1)
    tmp1 = sigma_cutoff*np.std(img1)
    img1[img1>tmp1] = tmp1
    return img1


#   FUNCTION generate_transform_xy
# Determines transformation matrices in x and y coordinates
#   INPUTS:
# img: currently used image (in cv2 format) to find transformation array of
# img_orig: image (in cv2 format) transformation array is based off of
# tfinit (default: None): base array passed into function
# offset_guess (default: [0,0]): Array showing initial estimate of distortion, in pixels
# warp_check_range (default: 10): distance (in pixels) that the function will search to find the
#     optimal transform matrix. Number of iterations = (warp_check_range+1)**2
# cumulative (default: False): determines if each image is compared to the previous image (default,
#      False), or to the original image (True). Output format is identical.
# cumulative_tform21 (default: np.eye(2,3,dtype=np.float32)): the transformation matrix, only used
#      if cumulative is switched to True.
#   OUTPUTS
# warp_matrix: transformation matrix used to convert img_orig into img

def generate_transform_xy(img, img_orig, tfinit=None, offset_guess = [0,0], warp_check_range=10, 
                          cumulative=False, cumulative_tform21=np.eye(2,3,dtype=np.float32)):
    # Here we generate a MOTION_EUCLIDEAN matrix by doing a 
    # findTransformECC (OpenCV 3.0+ only).
    # Returns the transform matrix of the img with respect to img_orig
    warp_mode = cv2.MOTION_TRANSLATION
    if tfinit is not None:
        warp_matrix = tfinit
    else:
        warp_matrix = np.eye(2,3,dtype=np.float32)
    number_of_iterations = 100000
    termination_eps = 1e-1#e-5
    term_flags = cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT
    
    if cumulative:
        offset_guess[0] = offset_guess[0]+cumulative_tform21[0,2]
        offset_guess[1] = offset_guess[1]+cumulative_tform21[1,2]

    criteria = (term_flags, number_of_iterations, termination_eps)

    diff = np.Inf
    offset1=0
    offset2=0
    for i in range(-warp_check_range//2,(warp_check_range//2)+1):
        for j in range(-warp_check_range//2,(warp_check_range//2)+1):
            warp_matrix[0,2] = 2*i + offset_guess[0]
            warp_matrix[1,2] = 2*j + offset_guess[1]
            try:
                (cc, tform21) = cv2.findTransformECC(img_orig, img, warp_matrix, warp_mode,
                                                     criteria)
                img_test = cv2.warpAffine(img, tform21, (512,512), flags=cv2.INTER_LINEAR +
                                          cv2.WARP_INVERSE_MAP);
                currDiff = np.sum(np.square(img_test[150:-150, 150:-150]
                                            -img_orig[150:-150, 150:-150]))
                if currDiff < diff:
                    diff = currDiff
                    offset1 = tform21[0,2]
                    offset2 = tform21[1,2]
            except:
                pass
            warp_matrix[0,2] = offset1
            warp_matrix[1,2] = offset2
    return warp_matrix


#   FUNCTION distortion_correction
# Applies distortion correction parameters to an image. The distortion corrected data is then
# cropped to show only the common data, or expanded to show the maximum extent of all possible data.
#   INPUTS:
# filename: name of hdf5 file containing data
# data_folder (default: 'datasets'): folder searched for image inputs
# selection (default: None): determines the name of folders or files to be used.
# criteria (default: None): determines what category selection refers to
# dm_data_folder (default: 'process/01-distortion_params'): folder searched for distoriton params
# dm_selection (default: None): determines the name of folders or files to be used.
# dm_criteria (default: None): determines what category selection refers to
# cropping (default: True): If set to True, each dataset is cropped to show only the common area. If
#     set to false, expands data shape to show all data points of all images.
#   OUTPUTS
# null

def distortion_correction_(filename, all_input_criteria, cropping = True):
    all_in_path_list = pt.path_search(filename, all_input_criteria, repeat='block')
    in_path_list = all_in_path_list[0]
    dm_path_list = all_in_path_list[1]
    
    distortion_matrices = []
    with h5py.File(filename, "a") as f:
        for path in dm_path_list[:]:
            distortion_matrices.append(np.copy(f[path]))
        xoffsets = []
        yoffsets = []
        for matrix in distortion_matrices:
            xoffsets.append(np.array(matrix[0,2]))
            yoffsets.append(np.array(matrix[1,2]))
    offset_caps = [np.max(xoffsets), np.min(xoffsets), np.max(yoffsets), np.min(yoffsets)]

    out_folder_locations = pt.find_output_folder_location(filename, 'distortion_correction',
                                                          in_path_list)

    with h5py.File(filename, "a") as f:
        start_time = time.time()
        for i in range(len(in_path_list)):
            orig_image = f[in_path_list[i]]
            if cropping == True:
                final_image = array_cropped(orig_image, xoffsets[i], yoffsets[i], offset_caps)
            else:
                final_image = array_expanded(orig_image, xoffsets[i], yoffsets[i], offset_caps)
            data = pt.write_output_f(f, final_image, out_folder_locations[i], [in_path_list[i],
                                                                               dm_path_list[i]])
            propagate_scale_attrs(data, f[in_path_list[i]])
            pt.progress_report(i+1, len(in_path_list), start_time, 'distortion_correction',
                            in_path_list[i])

            

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
        new_data.attrs['scale_m_per_px'] = old_data.attrs['scale_m_per_px']
    else:
        if ('size' in old_data.attrs) and ('shape' in old_data.attrs):
            scan_size = old_data.attrs['size']
            shape = old_data.attrs['shape']
            new_data.attrs['scale_m_per_px'] = scan_size[0] / shape[0]

            
#def distortion_correction_(filename, data_folder='datasets', selection=None, criteria=None,
#                           dm_data_folder = 'process/01-distortion_params', dm_selection=None,
#                           dm_criteria=None, cropping = True):
#    
#    dm_path_list = pt.path_inputs(filename, dm_data_folder, dm_selection, dm_criteria)
#    distortion_matrices = []
#    with h5py.File(filename, "a") as f:
#        for path in dm_path_list[:]:
#            distortion_matrices.append(np.copy(f[path]))
#        xoffsets = []
#        yoffsets = []
#        for matrix in distortion_matrices:
#            xoffsets.append(np.array(matrix[0,2]))
#            yoffsets.append(np.array(matrix[1,2]))
#    offset_caps = [np.max(xoffsets), np.min(xoffsets), np.max(yoffsets), np.min(yoffsets)]
#
#    in_path_list = pt.path_inputs(filename, data_folder, selection, criteria)
#    out_folder_locations = pt.find_output_folder_location(filename, 'distortion_correction',
#                                                          in_path_list)
#    if len(in_path_list)%len(dm_path_list):
#       print('Error: Images to be corrected are not a multiple of the amount of distortion\
#                matrices')
#        return

#    number_of_images_for_each_matrix = len(in_path_list)//len(dm_path_list)
#    with h5py.File(filename, "a") as f:
#        j = -1
#        start_time = time.time()
#        for i in range(len(in_path_list)):
#            if i%number_of_images_for_each_matrix == 0:
#                j = j+1
#            orig_image = f[in_path_list[i]]
#            if cropping == True:
#                final_image = array_cropped(orig_image, xoffsets[j], yoffsets[j], offset_caps)
#            else:
#                final_image = array_expanded(orig_image, xoffsets[j], yoffsets[j], offset_caps)
#            data = pt.write_output_f(f, final_image, out_folder_locations[i], [in_path_list[i],
#                                                                               dm_path_list[j]])
#            propagate_scale_attrs(data, f[in_path_list[i]])
#            pt.progress_report(i+1, len(in_path_list), start_time, 'distortion_correction',
#                            in_path_list[i])
    
    
#   FUNCTION array_cropped
# crops a numpy_array given the offsets of the array, and the minimum and maximum offsets of a set,
# to include only valid data shared by all arrays
#   INPUTS:
# array: the array to be cropped
# xoffset: the xoffset of the array
# yoffset: the yoffset of the array
# offset_caps: a list of four entries. In order, these entries are the xoffset maximum, xoffset
# minimum, yoffset maximum, and yoffset minimum fo all arrays
#   OUTPUTS:
# cropped_array: the cropped array

def array_cropped(array, xoffset, yoffset, offset_caps):
    if offset_caps != [0,0,0,0]:
        left = int(np.ceil(offset_caps[0])-np.floor(xoffset))
        right = int(np.floor(offset_caps[1])-np.floor(xoffset))
        top = int(np.ceil(offset_caps[2])-np.floor(yoffset))
        bottom = int(np.floor(offset_caps[3])-np.floor(yoffset))
        if right == 0:
            right = np.shape(array)[1]
        if bottom == 0:
            bottom = np.shape(array)[0]
        cropped_array = array[top:bottom, left:right]
    else:
        cropped_array = array
    return cropped_array


#   FUNCTION array_expanded
# expands a numpy_array given the offsets of the array, and the minimum and maximum offsets of a
# set, to include all points of each array. Empty data is set to be NaN
#   INPUTS:
# array: the array to be expanded
# xoffset: the xoffset of the array
# yoffset: the yoffset of the array
# offset_caps: a list of four entries. In order, these entries are the xoffset maximum, xoffset
# minimum, yoffset maximum, and yoffset minimum fo all arrays
#   OUTPUTS:
# expanded_array: the expanded array

def array_expanded(array, xoffset, yoffset, offset_caps):
    height = int(np.shape(array)[0]+np.ceil(offset_caps[2])-np.floor(offset_caps[3]))
    length = int(np.shape(array)[1]+np.ceil(offset_caps[0])-np.floor(offset_caps[1]))
    expanded_array = np.empty([height, length])
    expanded_array[:] = np.nan
    left = int(-np.floor(offset_caps[1])+xoffset)
    right = int(length-np.ceil(offset_caps[0])+xoffset)
    top = int(-np.floor(offset_caps[3])+yoffset)
    bottom= int(height-np.ceil(offset_caps[2])+yoffset)
    expanded_array[top:bottom, left:right] = array
    return expanded_array


#   FUNCTION phase_linearisation_
# Converts each entry of a 2D phase channel (rotating 360 degrees with an arbitrary 0 point) into a
# float between 0 and 1.  The most common values become 0 or 1, and other values are a linear
# interpolation between these two values. 0 and 1 are chosen such that the mean of the entire 
# channel does not become greater than a value defined by flip_proportion, and such that the 
# edgemost pixels are more 0 than the centre.
#   INPUTS:
# filename: name of hdf5 file containing data
# data_folder (default: 'datasets'): folder searched for image inputs
# selection (default: ['Phase1Trace', 'Phase1Retrace', 'Phase2Trace', 'Phase2Retrace']): determines
#     the name of folders or files to be used.
# criteria (default: channel): determines what category selection refers to
# min_separation (default: 90): minimum distance between the two peaks assigned as 0 and 1.
# background (default: None): number to identify where background is to correctly attribute values.
#     If positive, tries to make everything to the left of this value background; if negative, makes
#     everything to the right background
# flip_proportion (default: 0.8): threshold, above which the data is flipped to (1-data)
# print_frequency (default: 4): number of channels processed before a a status update is printed
# show (default: False): If True, show the data prior to saving
#   OUTPUTS
# null

def phase_linearisation_(filename, data_folder='datasets', selection = ['Phase1Trace',
                                                                        'Phase1Retrace',
                                                                        'Phase2Trace',
                                                                        'Phase2Retrace'],
                         criteria = 'channel', min_separation=90, background = None,
                         flip_proportion = 0.8, print_frequency = 4, show = False):
    in_path_list = pt.path_inputs(filename, data_folder, selection, criteria)
    out_folder_locations = pt.find_output_folder_location(filename, 'phase_linearisation',
                                                          in_path_list)
    with h5py.File(filename, "a") as f:
        for index in range(len(in_path_list)):
            path = in_path_list[index]
            phase_flat = np.array(f[path]).ravel()
            min_phase = int(np.floor(np.min(phase_flat)))
            max_phase = min_phase+360
            
            # Convert original data into histograms and find largest peak
            ydata, bin_edges = np.histogram(phase_flat, bins=360, range=[min_phase, max_phase])
            peak1_index = np.argmax(ydata)
            
            # Find next largest peak a distance away from original peak
            peak1_exclude_left = wrap(peak1_index-min_separation, 0,360) 
            peak1_exclude_right = wrap(peak1_index+min_separation, 0,360)
            if peak1_exclude_left < peak1_exclude_right:
                peak2_search_region = np.delete(ydata,
                                                np.arange(peak1_exclude_left,peak1_exclude_right))
                peak2_index = np.argmax(peak2_search_region)
                if peak2_index < peak1_exclude_left:
                    pass
                else:
                    peak2_index = peak2_index + 2*min_separation
            else:
                peak2_search_region = ydata[peak1_exclude_right:peak1_exclude_left]
                peak2_index = np.argmax(peak2_search_region) + peak1_exclude_right -1
            
            # Split wrapped dataset into two number lines; one going up and one down
            if peak1_index > peak2_index:
                peak1_index, peak2_index = peak2_index, peak1_index
            peak1_value = peak1_index+min_phase
            peak2_value = peak2_index+min_phase
            range_1to2 = peak2_value-peak1_value
            range_2to1 = 360 - range_1to2
            
            # Create a new array whose values depend on their position on the number lines
            linearised_array = np.copy(f[path])
            linearise_map = np.vectorize(linearise)
            linearised_array = linearise_map(linearised_array, peak1_value, peak2_value,
                                             range_1to2, range_2to1)
            # Define which points are 0 or 1 based on relative magnitude
            if np.mean(linearised_array) > flip_proportion:
                linearised_array = 1-linearised_array
            elif np.mean(linearised_array) > 1-flip_proportion:
                if background is None:
                    if (np.mean(linearised_array[:,:10])+np.mean(linearised_array[:,-10:])) \
                          > 2*np.mean(linearised_array):
                        linearised_array = 1-linearised_array
                elif background < 0:
                    if np.mean(linearised_array[:,background:]) > np.mean(linearised_array):
                        linearised_array = 1-linearised_array
                elif background > 0:
                    if np.mean(linearised_array[:,:background]) > np.mean(linearised_array):
                        linearised_array = 1-linearised_array
            pt.intermediate_plot(linearised_array, force_plot = show, text = 'Linearised Array')
            data = pt.write_output_f(f, medfilt(cv2.blur(linearised_array, (7,7)),7),
                                     out_folder_locations[index], in_path_list[index])
            data.attrs['peak values'] = [peak1_value, peak2_value]
            propagate_scale_attrs(data, f[path])
            if (index+1)%print_frequency == 0:
                print('Phase Linearisation: ' + str(index+1) + ' of ' + str(len(in_path_list))
                      + ' complete.')

                
#   FUNCTION phase_linearisation
# As phase_linearisation, but designed for use with pt.l_apply.
# Converts each entry of a 2D phase channel (rotating 360 degrees with an arbitrary 0 point) into a
# float between 0 and 1.  The most common values become 0 or 1, and other values are a linear
# interpolation between these two values. 0 and 1 are chosen such that the mean of the entire 
# channel does not become greater than a value defined by flip_proportion, and such that the 
# edgemost pixels are more 0 than the centre.
#   INPUTS:
# image: array that contains data
# min_separation (default: 90): minimum distance between the two peaks assigned as 0 and 1.
# background (default: None): number to identify where background is to correctly attribute values.
#     If positive, tries to make everything to the left of this value background; if negative, makes
#     everything to the right background
# flip_proportion (default: 0.8): threshold, above which the data is flipped to (1-data)
# print_frequency (default: 4): number of channels processed before a a status update is printed
# show (default: False): If True, show the data prior to saving
#   OUTPUTS
# result: dict containing data and attributes
                
def phase_linearisation(image, min_separation=90, background = None,
                         flip_proportion = 0.8, show = False):
    phase_flat = np.array(image).ravel()
    min_phase = int(np.floor(np.min(phase_flat)))
    max_phase = min_phase+360

    # Convert original data into histograms and find largest peak
    ydata, bin_edges = np.histogram(phase_flat, bins=360, range=[min_phase, max_phase])
    peak1_index = np.argmax(ydata)

    # Find next largest peak a distance away from original peak
    peak1_exclude_left = wrap(peak1_index-min_separation, 0,360) 
    peak1_exclude_right = wrap(peak1_index+min_separation, 0,360)
    if peak1_exclude_left < peak1_exclude_right:
        peak2_search_region = np.delete(ydata,
                                        np.arange(peak1_exclude_left,peak1_exclude_right))
        peak2_index = np.argmax(peak2_search_region)
        if peak2_index < peak1_exclude_left:
            pass
        else:
            peak2_index = peak2_index + 2*min_separation
    else:
        peak2_search_region = ydata[peak1_exclude_right:peak1_exclude_left]
        peak2_index = np.argmax(peak2_search_region) + peak1_exclude_right -1

    # Split wrapped dataset into two number lines; one going up and one down
    if peak1_index > peak2_index:
        peak1_index, peak2_index = peak2_index, peak1_index
    peak1_value = peak1_index+min_phase
    peak2_value = peak2_index+min_phase
    range_1to2 = peak2_value-peak1_value
    range_2to1 = 360 - range_1to2

    # Create a new array whose values depend on their position on the number lines
    linearised_array = np.copy(image)
    linearise_map = np.vectorize(linearise)
    linearised_array = linearise_map(linearised_array, peak1_value, peak2_value,
                                     range_1to2, range_2to1)
    # Define which points are 0 or 1 based on relative magnitude
    if np.mean(linearised_array) > flip_proportion:
        linearised_array = 1-linearised_array
    elif np.mean(linearised_array) > 1-flip_proportion:
        if background is None:
            if (np.mean(linearised_array[:,:10])+np.mean(linearised_array[:,-10:])) \
                  > 2*np.mean(linearised_array):
                linearised_array = 1-linearised_array
        elif background < 0:
            if np.mean(linearised_array[:,background:]) > np.mean(linearised_array):
                linearised_array = 1-linearised_array
        elif background > 0:
            if np.mean(linearised_array[:,:background]) > np.mean(linearised_array):
                linearised_array = 1-linearised_array
    pt.intermediate_plot(linearised_array, force_plot = show, text = 'Linearised Array')
    linearised = medfilt(cv2.blur(linearised_array, (7,7)),7)
    result = pt.hdf5_dict(linearised, peak_values=[peak1_value, peak2_value])
    return result
                
                
#   FUNCTION linearise
# Converts a phase entry (rotating 360 degrees with an arbitrary 0 point) into a float between 0 
# and 1, given the values of the two extremes, and the ranges between them
#   INPUTS:
# entry: the phase entry to be converted
# peak1_value: the phase value that would be linearised to 0
# peak2_value: the phase value that would be linearised to 1
# range_1to2: the distance in phase from peak1 to peak2
# range_1to2: the distance in phase from peak2 to peak1
#   OUTPUTS
# entry: the phase entry that has been converted
                
def linearise(entry, peak1_value, peak2_value, range_1to2, range_2to1):
    if (entry >= peak1_value) and (entry < peak2_value):
        entry = (entry-peak1_value)/range_1to2
    elif entry < peak1_value:
        entry = (peak1_value-entry)/range_2to1
    else:
        entry = 1-((entry-peak2_value)/range_2to1)
    return entry
                
                
#   FUNCTION sum_
# Adds multiple channels together. The files are added in order, first by channel and then by
# sample. The amount of source files in each destination file defined by entry_count.
#   INPUTS:
# filename: name of hdf5 file containing data
# data_folder (default: 'datasets'): folder searched for image inputs
# selection (default: None): determines the name of folders or files to be used.
# criteria (default: None): determines what category selection refers to
# entry_count (default: None): the amount of source files added together to each destination value.
#     Can be set to an int, or left as default to add all files together.
# output_name (default: None): the name of the final output file. By default, uses filename and the
#     suffix '_sum'
# folder_name (default: None): the folder ('sample') name the data files are added to. By default,
#     copies existing folder names for new folder names, and places files in the same folder name
#     as the last file summed.
#   OUTPUTS
# null      
                
def sum_(filename, data_folder='datasets', selection = None, criteria = None, entry_count = None,
         output_name = None, folder_name = None):
    in_path_list = pt.path_inputs(filename, data_folder, selection, criteria)
    if folder_name is None:
        folder_name = []
        for path in in_path_list:
            folder_name.append(path.split('.')[0])
    out_folder_locations = pt.find_output_folder_location(filename, 'sum', folder_name)
    with h5py.File(filename, "a") as f:
        if entry_count is None:
            entry_count = len(in_path_list)
        if output_name is None:
            output_name = filename+'_sum'
        for i in range(len(in_path_list)):
            path = in_path_list[i]
            curr_array = np.copy(f[path])
            curr_array = curr_array.astype(float)
            if i % entry_count == 0:
                sum_array = curr_array
                source_list = [path]
            else:
                sum_array = sum_array + curr_array
                source_list.append(path)
            if i % entry_count == entry_count-1:
                data = pt.write_output_f(f, sum_array, out_folder_locations[i-(entry_count-1)],
                                         source_list, output_name)
                propagate_scale_attrs(data, f[path])

#   FUNCTION m_sum_
# Adds multiple channels together. The files are added in order, first by channel and then by
# sample. The amount of source files in each destination file defined by entry_count. Replaces sum_
#   INPUTS:
# *args: arrays to be summed
#   OUTPUTS
# result: dict containing data and attributes
                
def m_sum(*args):
    total = 0
    for arg in args:
        total = total+arg
    input_count = len(args)
    result = pt.hdf5_dict(total, input_count=input_count)
    return result

#   FUNCTION phase_binarisation_
# Converts each entry of an array that is between two values to either 0 or 1. Designed for use
# with linearised phase data, where peaks exist at endpoints.
#   INPUTS:
# filename: name of hdf5 file containing data
# data_folder (default: 'datasets'): folder searched for image inputs
# selection (default: None): determines the name of folders or files to be used.
# criteria (default: channel): determines what category selection refers to
# thresh_estimate (default: 2): initial guess for where the threshold should be placed
# thresh_search_range (default: 0.8): range of thresholds searched around the estimate
# keep_name (default: False): if set to True, the data channel will maintain the same name as its
#      source. Otherwise, the data channel is renamed to BinarisedPhase
# print_frequency (default: 4): number of channels processed before a a status update is printed
#    OUTPUTS
# null
                
def phase_binarisation_(filename, data_folder='datasets', selection = None, criteria = None,
                        thresh_estimate = 2, thresh_search_range = 0.4, keep_name = False,
                        print_frequency = 4):
    in_path_list = pt.path_inputs(filename, data_folder, selection, criteria)
    out_folder_locations = pt.find_output_folder_location(filename, 'phase_binarisation',
                                                          in_path_list)
    with h5py.File(filename, "a") as f:
        for i in range(len(in_path_list)):
            path = in_path_list[i]
            phase = np.copy(f[path])
            blurred_phase = cv2.blur(phase, (7,7))
            best_thresh = threshold_noise(blurred_phase, thresh_estimate, thresh_search_range/2, 5)
            binary = blurred_phase > best_thresh

            if np.mean(binary) > 0.95:
                binary = 1-binary
            if keep_name:
                data = pt.write_output_f(f, binary, out_folder_locations[i], in_path_list[i])
            else:
                data = pt.write_output_f(f, binary, out_folder_locations[i], in_path_list[i],
                                         'BinarisedPhase')
            data.attrs['threshold'] = best_thresh
            propagate_scale_attrs(data, f[path])
            if (i+1)%print_frequency == 0:
                print('Binarisation: ' + str(i+1) + ' of ' + str(len(in_path_list)) + ' complete.')

                
def phase_binarisation (phase, thresh_estimate = None, thresh_search_range = None, source_input_count = None):
    if thresh_estimate is None:
        if source_input_count is not None:
            thresh_estimate = source_input_count/2
        else:
            thresh_estimate = 0.5
    if thresh_search_range is None:
        if source_input_count is not None:
            thresh_search_range = source_input_count/10
        else:
            thresh_search_range = 0.1
    blurred_phase = cv2.blur(phase, (7,7))
    best_thresh = threshold_noise(blurred_phase, thresh_estimate, thresh_search_range/2, 5)
    binary = blurred_phase > best_thresh

    if np.mean(binary) > 0.95:
        binary = 1-binary
    result = pt.hdf5_dict(binary, threshold=best_thresh)
    return result
                
#   FUNCTION threshold_noise
# Iterative threshold function designed for phase_binarisation). Decides threshold based on what
# gives the "cleanest" image, with minimal high frequency noise.
#   INPUTS:
# image: data to be thresholded
# old_guess: initial estimate of threshold
# thresh_range: span searched (in both positive and negative directions) for optimal threshold
# iterations: number of times the function is run iteratively
# old_diff (default: None): number that represents the number of noise. Determines best threshold.
#   OUTPUTS
# best_guess: final guess for best threshold

def threshold_noise(image, old_guess, thresh_range, iterations, old_diff = None):
    if old_diff is None:
        binary = image>old_guess
        binary_filt = medfilt(binary, 3)
        #binary_filt = contour_closure(binary_filt)
        old_diff = np.sum(np.abs(binary_filt-binary))
    if iterations > 0:
        new_guesses = [old_guess-thresh_range, old_guess+thresh_range]
        diffs = []
        for thresh in new_guesses:
            binary = image>thresh
            binary_filt = medfilt(binary, 3)
            #binary_filt = contour_closure(binary_filt)
            diffs.append(np.sum(np.abs(binary_filt-binary)))
        best_i = np.argmin(diffs)
        best_guess = threshold_noise(image, new_guesses[best_i], thresh_range/2, iterations-1,
                                     diffs[best_i])
    else:
        best_guess = old_guess
    return best_guess
                
    
#   FUNCTION wrap
# Extended modulo function. Converts a number outside of a range between two numbers by continually
# adding or substracting the span between these numbers.
#   INPUTS:
# x: number to be wrapped
# low (default: 0): lowest value of the wrap
# high (default: 0): highest value of the wrap
#   OUTPUTS
# x: wrapped number

def wrap(x, low = 0, high = 360):
    angle_range = high-low
    while x <= low:
        x = x+angle_range
    while x > high:
        x = x-angle_range
    return x
            

#   FUNCTION contour_closure_
# Removes regions on a binarised image with an area less than a value defined by size_threshold.
# This is performed by finding the contours and the area of the contours, thus providing no change
# to the bulk of the image itself (as a morphological closure would)
#   INPUTS:
# filename: name of hdf5 file containing data
# data_folder (default: 'datasets'): folder searched for image inputs
# selection (default: ['Amplitude1Retrace', 'Amplitude2Retrace']): determines the name of folders or
#     files to be used.
# criteria (default: channel): determines what category selection refers to
# size_threshold (default: 100): area in pixels that a contour is compared to before being closed
# type_bool (default: True): sets data to a boolean type
#   OUTPUTS
# null

def contour_closure_(filename, data_folder = 'datasets', selection = None, criteria = None, 
                     size_threshold = 100, type_bool = True):
    in_path_list = pt.path_inputs(filename, data_folder, selection, criteria)
    out_folder_locations = pt.find_output_folder_location(filename, 'contour_closure',
                                                              in_path_list)
    with h5py.File(filename, "a") as f:
        for i in range(len(in_path_list)):
            image = contour_closure(f[in_path_list[i]],
                                    size_threshold = size_threshold, type_bool = type_bool)
            data=pt.write_output_f(f, image, out_folder_locations[i], in_path_list[i])
            data.attrs['size_threshold'] = size_threshold
            propagate_scale_attrs(data, f[in_path_list[i]])    
        
#   FUNCTION contour_closure
# Removes regions on a binarised image with an area less than a value defined by size_threshold.
# This is performed by finding the contours and the area of the contours, thus providing no change
# to the bulk of the image itself (as a morphological closure would)
#   INPUTS:
# source: image to be closed
# size_threshold (default: 100): area in pixels that a contour is compared to before being closed
# type_bool (default: True): sets data to a boolean type
#   OUTPUTS
# null

def contour_closure(source, size_threshold = 50, type_bool = True):
    source = np.array(source).astype('uint8')
    image = np.zeros_like(source)
    cv2_image,contours,hierarchy = cv2.findContours(source,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    new_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > size_threshold:
            new_contours.append(contour)
    cv2.drawContours(image,new_contours,-1, (1), thickness=cv2.FILLED)
    if type_bool:
        image = image.astype(bool)
    return image
        
#   FUNCTION find_a_domains_
# Determines the a-domains in an amplitude image, by looking for points of high second derivative.
# These points are then fit to lines, and these lines filtered to the most common lines that are 
# either parallel or perpendicular to one another. If given, the phase data can also be used to
# distinguish between a-domains and 180 degree walls.
#   INPUTS:
# filename: name of hdf5 file containing data
# data_folder (default: 'datasets'): folder searched for image inputs
# selection (default: ['Amplitude1Retrace', 'Amplitude2Retrace']): determines the name of folders or
#     files to be used.
# criteria (default: channel): determines what category selection refers to
# pb_data_folder (default: None): folder searched for phase_binarisation. If none, no filter will be
#     made or used
# pb_selection (default: None): determines the name of folders or files to be used.
# pb_criteria (default: channel): determines what category selection refers to
# direction (default: None): Direction of the a domains found:
#     None: Finds domain walls at any angle
#     'Vert': Finds vertical domain walls
#     'Horz': Finds horizontal domain walls
# filter_width (default: 15): total width of the filter, in pixels, around the domain-wall
#     boundaries. This is the total distance - so half this value is applied to each side.
# thresh_factor (default: 2): factor used by binarisation. A higher number gives fewer valid points.
# dilation (default: 2): amount of dilation steps to clean image
# erosion (default: 4): amount of erosion steps to clean image
# line_threshold (default: 80): minimum number of votes (intersections in Hough grid cell)
# min_line_length (default: 80): minimum number of pixels making up a line
# max_line_gap (default: 80): maximum gap in pixels between connectable line segments
# plots (default: [None]): option to plot intermediary steps. Plots if the following are in array:
#     'amp': Raw amplitude data that contains a-domains
#     'phase': Binarised phase data
#     'filter': Filter made from the domain walls visible in phase
#     'spline': Spline fit of original amplitude data
#     'first_deriv': First derivitave of amplitude
#     'second_deriv': Second derivitave of amplitude
#     'binary': Binarisation of second derivative
#     'erode': Binarisation data after an erosion filter is applied
#     'lines': Lines found, and should correspond to a-domains on original amplitude image
#     'clean': Lines found, after filtering to the most common angles
# print_frequency (default: 4): number of channels processed before a a status update is printed
#   OUTPUTS
# null

def find_a_domains_(filename, data_folder = 'datasets', 
                    selection = ['Amplitude1Retrace', 'Amplitude2Retrace'],
                    criteria = 'channel', pb_data_folder = None, pb_selection = None,
                    pb_criteria = None, direction = None, filter_width = 15, thresh_factor = 2,
                    dilation = 2, erosion = 4, line_threshold = 50, min_line_length=50,
                    max_line_gap=10, plots = None, print_frequency = 4):
    in_path_list = pt.path_inputs(filename, data_folder, selection, criteria)
    if pb_data_folder is not None:
        pb_path_list = pt.path_inputs(filename, pb_data_folder, pb_selection, pb_criteria)
    out_folder_locations = pt.find_output_folder_location(filename, 'a_domains', in_path_list)
    
    rotation_list = []
    if pb_data_folder is not None:
        phase_filter = True
        if len(in_path_list)%len(pb_path_list):
            print('Error: Images to be corrected are not a multiple of the amount of\
                  distortion matrices')
            return
        amplitudes_per_phase = len(in_path_list)//len(pb_path_list)
    else:
        phase_filter = False
    with h5py.File(filename, "a") as f:
        for index in range(len(in_path_list)):
            path = in_path_list[index]
            
            if plots is not None:
                print('-----')
                print(path)       
        
            if phase_filter:
                pb_index = int(np.floor(index/amplitudes_per_phase))
                pb_path = pb_path_list[pb_index]
                domain_wall_filter = create_domain_wall_filter(f[pb_path],
                                                               filter_width = filter_width,
                                                               plots = plots)
            else:
                domain_wall_filter = np.zeros_like(f[path])+1
            a_estimate, bin_thresh = estimate_a_domains(f[path], domain_wall_filter,
                                                        direction = direction,
                                                        plots = plots,
                                                        thresh_factor = thresh_factor,
                                                        dilation = dilation, erosion = erosion)

            # Find Lines
            rho = 1  # distance resolution in pixels of the Hough grid
            theta = np.pi / 180  # angular resolution in radians of the Hough grid
            line_image = np.copy(a_estimate) * 0  # creating a blank to draw lines on

            # Run Hough on edge detected image
            # Output "lines" is an array containing endpoints of detected line segments
            lines = cv2.HoughLinesP(a_estimate, rho, theta, line_threshold, np.array([]),
                                    min_line_length, max_line_gap)
            
            if lines is not None:
                # Draw lines, filtering with phase filter if possible
                phase_filter_lines = []
                for line in lines:
                    for x1,y1,x2,y2 in line:
                        if pb_data_folder is not None:
                            blank = np.zeros_like(line_image)
                            one_line = cv2.line(blank,(x1,y1),(x2,y2),(255,0,0),5)
                            points_outside_mask = one_line*domain_wall_filter
                            if np.sum(points_outside_mask) > 0.2*np.sum(one_line):
                                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
                                phase_filter_lines.append(line)
                        else:
                            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
                            phase_filter_lines.append(line)
                lines_edges = cv2.addWeighted(a_estimate, 0.8, line_image, 1, 0)
                pt.intermediate_plot(line_image, 'lines', plots, 'Lines Found')

                # Find angles of each line
                angles = []
                for line in phase_filter_lines:
                    for x1,y1,x2,y2 in line:
                        if x2 == x1:
                            angles.append(90)
                        else:
                            angles.append((180*np.arctan((y2-y1)/(x2-x1))/np.pi))
                
                # Find first angle guess
                if direction == 'Vert':
                    key_angles = [-90, 90]
                elif direction == 'Horz':
                    key_angles = [0, 180]
                else:
                    key_angles = find_desired_angles(angles)

                # Filter To Angle-Valid Lines
                angle_filter_lines = []
                i = 0
                for angle in angles:
                    for key_angle in key_angles:
                        if check_within_angle_range(angle, key_angle, 1) == True:
                            angle_filter_lines.append(phase_filter_lines[i])
                    i = i+1

                # Draw Lines
                line_image = np.copy(a_estimate) * 0  # creating a blank to draw lines on
                for line in angle_filter_lines:
                    for x1,y1,x2,y2 in line:
                        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
                lines_edges = cv2.addWeighted(a_estimate, 0.8, line_image, 1, 0)
                pt.intermediate_plot(line_image, 'clean', plots, 'Lines Found, after filtering')

            data = pt.write_output_f(f, line_image, out_folder_locations[index],
                                     [in_path_list[index], pb_path_list[pb_index]])
            data.attrs['binarisation_threshold'] = bin_thresh
            data.attrs['filter_width'] = filter_width
            data.attrs['line_threshold'] = line_threshold
            data.attrs['min_line_length'] = min_line_length
            data.attrs['max_line_gap'] = max_line_gap
            data.attrs['thresh_factor'] = thresh_factor
            propagate_scale_attrs(data, f[path])
            if (index+1)%print_frequency == 0:
                print('Finding a-Domains. Scan ' + str(index+1) + ' of ' + str(len(in_path_list))
                      + ' complete.')
                
                
#   FUNCTION create_domain_wall_filter
# Creates a filter from the phase binarisation data, which can be used to find a-domains
#   INPUTS:
# phase: Binarised phase image from which images are to be obtained
# filter_width (default: 15): total width of the filter, in pixels, around the domain-wall
#     boundaries. This is the total distance - so half this value is applied to each side.
# plots (default: []): option to plot intermediary steps. Plots if the following are in array:
#     'phase': Binarised phase data
#     'filter': Filter made from the domain walls visible in phase
#   OUTPUTS
# domain_wall_filter: filter made from phase binarisation data

def create_domain_wall_filter(phase, filter_width = 15, plots = []):
    # Binarised_Phase
    binPhase = np.copy(phase)
    pt.intermediate_plot(binPhase, 'phase', plots, 'Binarised Phase')
    
    # Create Filter
    ite = filter_width//2
    domain_wall_filter = ~(binary_dilation(binPhase, iterations = ite)^
                           binary_erosion(binPhase, iterations = ite, border_value=1))
    pt.intermediate_plot(domain_wall_filter, 'filter', plots, 'Domain Wall Filter')
    return domain_wall_filter


#   FUNCTION estimate_a_domains
# Refines an amplitude image to points of higher second derivative, which are likely to correspond
# to domain walls. If phase is given, this is used to further the filter the lines to find only
# the a-domains. Function allows for optional viewing of intermediate steps.
#   INPUTS:
# amplitude: Amplitude image from which domain walls are to be emphasised
# domain_wall_filter (default: None): Filter used during row alignment to ignore effects of 180
#     degree walls
# direction (default: None): Direction of the derivative taken:
#     None: Takes both derivatives, adding together the values of the second derivative.
#     'Vert': Finds vertical domain walls (differentiates horizontally)
#     'Horz': Finds horizontal domain walls (differentiates vertically)
# plots (default: []): option to plot intermediary steps. Plots if the following are in array:
#     'amp': Raw amplitude data that contains a-domains
#     'row_align': Data after row-alignment (if any)
#     'spline': Spline fit of original amplitude data
#     'first_deriv': First derivitave of amplitude
#     'second_deriv': Second derivitave of amplitude
#     'binary': Binarisation of second derivative
#     'erode': Binarisation data after an erosion filter is applied
# thresh_factor (default: 2): factor used by binarisation. A higher number gives fewer valid points.
# dilation (default: 2): amount of dilation steps to clean image
# erosion (default: 4): amount of erosion steps to clean image
#   OUTPUTS
# filtered_deriv_amp: adjusted amplitude image made to highlight points of higher second derivative
# thresh: the threshold value used to find the a-domains

def estimate_a_domains(amplitude, domain_wall_filter=None, direction = None, plots = [],
                       thresh_factor = 2, dilation = 2, erosion = 4):
    # Raw Data
    amp = np.copy(amplitude)
    pt.intermediate_plot(amp, 'amp', plots, 'Original Data')
        
    # Row Alignment, if direction set
    if direction == 'Vert':
        amp = align_rows(amp, domain_wall_filter)
    elif direction == 'Horz':
        amp = align_rows(amp, domain_wall_filter, cols = True)
    pt.intermediate_plot(amp, 'row_align', plots, 'Row Aligned Data')

    # Fit to a spline (reduce high frequency noise)
    spline_amp = cspline2d(amp, 2.0)
    pt.intermediate_plot(spline_amp, 'spline', plots, 'Spline Fitted Data')
            
    # Find derivatives to highlight peaks
    if direction == 'Vert':
        first_deriv = np.gradient(spline_amp)[1]
        pt.intermediate_plot(first_deriv, 'first_deriv', plots, 'First Derivatives')
        deriv_amp = (np.gradient(first_deriv))[1]
        pt.intermediate_plot(deriv_amp, 'second_deriv', plots, 'Second Derivatives')
    elif direction == 'Horz':
        first_deriv = np.gradient(spline_amp)[0]
        pt.intermediate_plot(first_deriv, 'first_deriv', plots, 'First Derivatives')
        deriv_amp = (np.gradient(first_deriv))[0]
        pt.intermediate_plot(deriv_amp, 'second_deriv', plots, 'Second Derivatives')
    else:
        if direction is not None:
            print('Direction should be set to either \'Vert\', \'Horz\' or None. Behaviour\
                    defaulting to None')
        first_deriv = np.gradient(spline_amp)
        pt.intermediate_plot(first_deriv[0]+first_deriv[1], 'first_deriv', plots,
                             'First Derivatives')
        second_deriv_y = np.gradient(first_deriv[0])[0]
        second_deriv_x = np.gradient(first_deriv[1])[1]
        deriv_amp = second_deriv_y+second_deriv_x
        pt.intermediate_plot(deriv_amp, 'second_deriv', plots, 'Second Derivatives')
    
    # Binarise second derivative
    thresh = threshold_after_peak(deriv_amp, thresh_factor)
    binary = (deriv_amp > thresh)
    pt.intermediate_plot(binary, 'binary', plots, 'Binarised Derivatives')
    
    # Remove Small Points
    filtered_deriv_amp = binary_erosion(binary_dilation(binary, iterations = dilation),
                                        iterations = erosion)
    pt.intermediate_plot(filtered_deriv_amp, 'erode', plots, 'Eroded Binary')
    return filtered_deriv_amp.astype(np.uint8), thresh
            
    
#   FUNCTION align_rows
# Aligns rows (or cols) of an array, with a mask provided
#   INPUTS:
# array: the array to be aligned
# mask (default: None): mask of data to be ignored when aligning rows
# cols (default: False): If set to true, the columns are instead aligned
#   OUTPUTS
# new_array: the row (or col) aligned array
    
def align_rows(array, mask = None, cols = False):
    if mask is None:
        mask = 1 + zeros_like(array)
    if cols:
        array = np.transpose(array)
        mask = np.transpose(mask)
    masked_array = np.copy(array)
    masked_array = np.where(mask == 0, np.nan, masked_array)
    new_array = np.zeros_like(array)
    for i in range(np.shape(array)[0]):
        if all(np.isnan(masked_array[i])) or (np.mean(mask[i])<0.05):
            new_array[i] = array[i]-np.nanmean(array[i])
        else:
            new_array[i] = array[i]-np.nanmean(masked_array[i])
    if cols:
        new_array = np.transpose(new_array)
    return new_array
    
    
#   FUNCTION threshold_after_peak
# Creates a threshold value, used when finding a-domains. This works by creatinga histogram of all
# valid values, and finding the maximum value of this histogram. The threshold is the point where
# the height of the maximum is less than the the maximum divided by the factor passed to this
# function.
#   INPUTS:
# deriv_amp: data passed in to obtain the threshold.
# thresh_factor (default: 2): The factor the maximum is divided by to find the threshold. A higher
#     number gives fewer valid points.
#   OUTPUTS
# thresh: the determined value of the optimal threshold
                
def threshold_after_peak(deriv_amp, factor = 4):
    deriv_hist = np.histogram(deriv_amp.ravel(), bins = 256)
    max_counts = np.max(deriv_hist[0])
    found_max = False
    i = 0
    for count in deriv_hist[0]:
        if count == max_counts:
            found_max = True
        if found_max == True and count < max_counts/factor:
            found_max = False
            thresh = deriv_hist[1][i]
        i = i+1
    return thresh

#   FUNCTION find_desired_angles
# Finds best angles to find a-domains, by sorting all angles into a histogram and finding the most
# common angle. A list of this angle, its antiparallel, and its two perpendiculars are then 
# returned.
#   INPUTS:
# raw_data: a list of angles to be given
#   OUTPUTS
# angles: A list of the four angles that the a-domains should fit to

def find_desired_angles(raw_data):
    angles = np.zeros(4)
    ydata, bin_edges = np.histogram(raw_data, bins=360, range=[-180, 180])
    base_angle = (np.argmax(ydata))-180
    for i in range(4):
        angles[i] = wrap(base_angle+90*i, -180, 180)
    return angles


#   FUNCTION check_within_angle_range
# Checks if an angle is within a valid range around another angle given. This function uses the wrap
# functionality to search a full revolution.
#   INPUTS:
# angle: one of the angles to be searched
# key_angle: another angle to be compared to
# angle_range: the range that angle and key_angle must be within one another
# low (default: -180): the minimum value of the angle span
# high (default: 180): the maximum value of the angle span
#   OUTPUTS
# status: a bool stating whether the angles are in range (True) or not (False)

def check_within_angle_range(angle, key_angle, angle_range, low = -180, high = 180):
    low_angle = wrap(key_angle-angle_range, low, high)
    high_angle = wrap(key_angle+angle_range, low, high)
    status = False
    if low_angle < angle:
        if angle < high_angle:
            status = True
    if low_angle > high_angle:
        if angle < high_angle:
            status = True
        elif angle > low_angle:
            status = True
    return status


#   FUNCTION find_a_domain_angle_
# Creates a transformation matrix that would rotate each image around the centre such that a-domains
# (or some other vertical feature) is oriented vertically and horizontally. Works by finding the
# a-domains (by looking for points of high second derivative), finding the most common angles in
# for these a-domains, and taking the median of these angles along all images
#   INPUTS:
# filename: name of hdf5 file containing data
# data_folder (default: 'datasets'): folder searched for image inputs
# selection (default: ['Amplitude1Retrace', 'Amplitude2Retrace']): determines the name of folders or
#     files to be used.
# criteria (default: channel): determines what category selection refers to
# pb_data_folder (default: None): folder searched for phase_binarisation. If none, no filter will be
#     made or used
# pb_selection (default: None): determines the name of folders or files to be used.
# pb_criteria (default: channel): determines what category selection refers to
# filter_width (default: 15): total width of the filter, in pixels, around the domain-wall
#     boundaries. This is the total distance - so half this value is applied to each side.
# thresh_factor (default: 2): factor used by binarisation. A higher number gives fewer valid points.
# dilation (default: 2): amount of dilation steps to clean image
# erosion (default: 4): amount of erosion steps to clean image
# line_threshold (default: 80): minimum number of votes (intersections in Hough grid cell)
# min_line_length (default: 80): minimum number of pixels making up a line
# max_line_gap (default: 80): maximum gap in pixels between connectable line segments
# plots (default: [None]): option to plot intermediary steps. Plots if the following are in array:
#     'amp': Raw amplitude data that contains a-domains
#     'phase': Binarised phase data
#     'filter': Filter made from the domain walls visible in phase
#     'spline': Spline fit of original amplitude data
#     'first_deriv': First derivitave of amplitude
#     'second_deriv': Second derivitave of amplitude
#     'binary': Binarisation of second derivative
#     'erode': Binarisation data after an erosion filter is applied
#     'lines': Lines found, and should correspond to a-domains on original amplitude image
# print_frequency (default: 4): number of channels processed before a a status update is printed
#    OUTPUTS
# null

def find_a_domain_angle_(filename, data_folder = 'datasets',
                         selection = ['Amplitude1Retrace', 'Amplitude2Retrace'],
                         criteria = 'channel', pb_data_folder = None, pb_selection = None,
                         pb_criteria = None, filter_width = 15, thresh_factor=2, dilation=2,
                         erosion=4,line_threshold=80, min_line_length=80, max_line_gap=80,
                         plots = None, print_frequency = 4):
    in_path_list = pt.path_inputs(filename, data_folder, selection, criteria)
    if pb_data_folder is not None:
        pb_path_list = pt.path_inputs(filename, pb_data_folder, pb_selection, pb_criteria)
    out_folder_locations = pt.find_output_folder_location(filename, 'rotation_params', in_path_list)
    
    rotation_list = []
    if pb_data_folder is not None:
        phase_filter = True
        if len(in_path_list)%len(pb_path_list):
            print('Error: Images to be corrected are not a multiple of the amount of\
                  distortion matrices')
            return
        amplitudes_per_phase = len(in_path_list)//len(pb_path_list)
    else:
        phase_filter = False
    with h5py.File(filename, "a") as f:
        start_time = time.time()
        for index in range(len(in_path_list)):
            path = in_path_list[index]
            
            if plots is not None:
                print('-----')
                print(path)       
        
            if phase_filter:
                pb_index = int(np.floor(index/amplitudes_per_phase))
                pb_path = pb_path_list[pb_index]
                domain_wall_filter = create_domain_wall_filter(f[pb_path],
                                                               filter_width = filter_width,
                                                               plots = plots)
            else:
                domain_wall_filter = np.zeros_like(f[path])+1
            a_estimate, bin_thresh = estimate_a_domains(f[path], domain_wall_filter,
                                                        plots = plots,
                                                        thresh_factor = thresh_factor,
                                                        dilation = dilation,
                                                        erosion = erosion)
            
            # Find Lines
            rho = 1  # distance resolution in pixels of the Hough grid
            theta = np.pi / 180  # angular resolution in radians of the Hough grid
            line_image = np.copy(a_estimate) * 0  # creating a blank to draw lines on

            # Run Hough on edge detected image
            # Output "lines" is an array containing endpoints of detected line segments
            lines = cv2.HoughLinesP(a_estimate, rho, theta, line_threshold, np.array([]),
                                    min_line_length, max_line_gap)
            
            if lines is not None:
                # Draw lines, filtering with phase filter if possible
                valid_lines = []
                for line in lines:
                    for x1,y1,x2,y2 in line:
                        if pb_data_folder is not None:
                            blank = np.zeros_like(line_image)
                            one_line = cv2.line(blank,(x1,y1),(x2,y2),(255,0,0),5)
                            points_outside_mask = one_line*domain_wall_filter
                            if np.sum(points_outside_mask) > 0.2*np.sum(one_line):
                                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
                                if (x1 != x2) and (y1 != y2):
                                    valid_lines.append(line)
                        else:
                            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
                            if (x1 != x2) and (y1 != y2):
                                valid_lines.append(line)
                lines_edges = cv2.addWeighted(a_estimate, 0.8, line_image, 1, 0)
                pt.intermediate_plot(line_image, 'lines', plots, 'Lines Found')

                # Find first angle guess
                angles = []
                for line in valid_lines:
                    for x1,y1,x2,y2 in line:
                        if x2 == x1:
                            angles.append(90)
                        else:
                            angles.append((180*np.arctan((y2-y1)/(x2-x1))/np.pi))
                key_angles = find_desired_angles(angles)
                
                # Refine angle estimate
                for repetitions in range(3):
                    valid_lines = []
                    angle_offsets = []
                    i = 0
                    for angle in angles:
                        for key_angle in key_angles:
                            if check_within_angle_range(angle, key_angle, 2.5) == True:
                                valid_lines.append(lines[i])
                                angle_offsets.append(angles[i]-key_angle)
                        i = i+1
                    key_angles = key_angles + np.mean(angle_offsets)
                rotation_deg = -key_angles[np.argmin(np.abs(key_angles))]
                rotation_list.append(rotation_deg)
                average_angle = np.mean(rotation_list)
            pt.progress_report(index+1, len(in_path_list), start_time, 'a_angle',
                            in_path_list[index])
        
        
        rotation_array = np.array(rotation_list)
        rotation_array = sorted(rotation_array[~np.isnan(rotation_array)])
        average_angle = rotation_array[int(len(rotation_array)/2)]
        orig_y, orig_x = (f[in_path_list[index]].attrs['shape'])
        warp_matrix=cv2.getRotationMatrix2D((orig_x/2, orig_y/2), average_angle, 1)
        data = pt.write_output_f(f, warp_matrix, out_folder_locations[0], in_path_list,
                                 filename.split('.')[0])
        data.attrs['angle offset (degs)'] = average_angle
        data.attrs['binarisation_threshold'] = bin_thresh
        data.attrs['filter_width'] = filter_width
        data.attrs['line_threshold'] = line_threshold
        data.attrs['min_line_length'] = min_line_length
        data.attrs['max_line_gap'] = max_line_gap
        data.attrs['thresh_factor'] = thresh_factor
        
    
#   FUNCTION rotation_alignment_
# Applies a rotation matrix to an image. An option also allows for cropping to the largest common
# area, which is found via trial and error. If one rotation matrix is given, it is applied to all
# images given. If multiple rotation matrices are given, it applies each rotation matrix n times
# consecutively, where n is the amount of images divided by the number of rotation matrices. If this
# would not be a whole number, the program returns an error and ends without running.
#   INPUTS:
# filename: name of hdf5 file containing data
# data_folder (default: 'datasets'): folder searched for image inputs
# selection (default: None): determines the name of folders or files to be used.
# criteria (default: None): determines what category selection refers to
# rm_data_folder (default: 'process/01-rotation_params'): folder searched for the rotation matrix
# rm_selection (default: None): determines the name of folders or files to be used.
# rm_criteria (default: None): determines what category selection refers to
# cropping (default: True): determines if the image should be cropped to the maximum common area.
#     If this value is set to False, the image will not be intentionally cropped and the image will
#     maintain consistent dimensions. This will often result in some cropping regardless.
#   OUTPUTS
# null
#   TO DO:
# Allow for true non-cropping, which would extend the border to the maximum possible limit.
        
def rotation_alignment_(filename, data_folder='datasets', selection=None,
                         criteria=None, rm_data_folder = 'process/01-rotation_params',
                         rm_selection=None, rm_criteria=None, cropping = True):
    rm_path_list = pt.path_inputs(filename, rm_data_folder, rm_selection, rm_criteria)
    rotation_matrices = []
    with h5py.File(filename, "a") as f:
        for path in rm_path_list[:]:
            rotation_matrices.append(np.copy(f[path]))

    in_path_list = pt.path_inputs(filename, data_folder, selection, criteria)
    out_folder_locations = pt.find_output_folder_location(filename, 'rotation_alignment',
                                                          in_path_list)
    if len(in_path_list)%len(rm_path_list):
        print('Error: Images to be corrected are not a multiple of the amount of\
              distortion matrices')
        return
    number_of_images_for_each_matrix = len(in_path_list)//len(rm_path_list)
    with h5py.File(filename, "a") as f:
        start_time = time.time()
        j = -1
        for i in range(len(in_path_list)):
            if i%number_of_images_for_each_matrix == 0:
                j = j+1
            orig_img = np.copy(f[in_path_list[i]])
            
            orig_y, orig_x = (f[in_path_list[i]].attrs['shape'])
            
            if orig_img.dtype == bool:
                array_is_bool = True
                orig_img = orig_img.astype(float)
            else:
                array_is_bool = False
            
            new_img = cv2.warpAffine(orig_img, rotation_matrices[j], (orig_x,orig_y),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                     borderValue = np.nan);
            
            largest_area = 0
            if cropping:
                if i == 0:
                    for top in range(int(np.floor(orig_y/2))):
                        width_left = 0
                        for x in range(len(new_img[top])):
                            if np.isnan(new_img[top][x]) == False:
                                if width_left == 0:
                                    width_left = (orig_x/2)-x
                                width_right = x+1-(orig_x/2)
                        height = (orig_y/2)-top
                        width = np.min([width_left, width_right])
                        area = height*width
                        if area > largest_area:
                            largest_area = area
                            best_top = top
                            best_left = int((orig_x/2)-width)
                new_img = new_img[best_top:orig_y-best_top, best_left:orig_x-best_left]
                while np.isnan(sum(new_img[-1])):
                    new_img = new_img[:-1]
            
            if array_is_bool:
                new_img = new_img.astype(bool)
            
            data = pt.write_output_f(f, new_img, out_folder_locations[i], [in_path_list[i],
                                                                           rm_path_list[j]])
            propagate_scale_attrs(data, f[in_path_list[i]])
            pt.progress_report(i+1, len(in_path_list), start_time, 'a_alignment',
                            in_path_list[i])

            
#   FUNCTION threshold_
# Thresholds an image by passing in a ratio between the minimum and maximum values of this image
#   INPUTS:
# filename: name of hdf5 file containing data
# data_folder (default: 'datasets'): folder searched for image inputs
# selection (default: None): determines the name of folders or files to be used.
# criteria (default: None): determines what category selection refers to
# thresh_ratio (default: 0.5): ratio between the minimum and maximum of the image to threshold
# overwrite (default: False): if set to True, if this function was the last process run, the last
#     run will be overwritten and replaced with this. To be used sparingly, and only if function
#     parameters must be guessed and checked
#   OUTPUTS
# null

def threshold_(filename, data_folder = 'datasets', selection = None, criteria = None,
               thresh_ratio = 0.5, overwrite = False):
    in_path_list = pt.path_inputs(filename, data_folder, selection, criteria)
    out_folder_locations = pt.find_output_folder_location(filename, 'threshold', in_path_list,
                                                          overwrite)
    with h5py.File(filename, "a") as f:
        for index in range(len(in_path_list)):
            path = in_path_list[index]
            max_level = np.nanmax(f[path])
            min_level = np.nanmin(f[path])
            real_threshold = min_level+thresh_ratio*(max_level-min_level)
            thresh_data = f[path]>real_threshold
            data = pt.write_output_f(f, thresh_data, out_folder_locations[index],
                                     in_path_list[index], output_name = 'BinarisedADomains')
            data.attrs['threshold'] = real_threshold
            data.attrs['thresh ratio'] = thresh_ratio
            propagate_scale_attrs(data, f[in_path_list[index]])
            
            
#   FUNCTION skeletonize_
# Applies the skeletonize function from skimage.morphology
#   INPUTS:
# filename: name of hdf5 file containing data
# data_folder (default: 'datasets'): folder searched for image inputs
# selection (default: None): determines the name of folders or files to be used.
# criteria (default: None): determines what category selection refers to
#   OUTPUTS
# null

def skeletonize_(filename, data_folder = 'datasets', selection = None, criteria = None):
    in_path_list = pt.path_inputs(filename, data_folder, selection, criteria)
    out_folder_locations = pt.find_output_folder_location(filename, 'skeletonize', in_path_list)
    with h5py.File(filename, "a") as f:
        for index in range(len(in_path_list)):
            path = in_path_list[index]
            raw_data = np.copy(f[path])
            skeleton_data = skeletonize(raw_data)
            data = pt.write_output_f(f, skeleton_data, out_folder_locations[index],
                                     in_path_list[index])
            propagate_scale_attrs(data, f[in_path_list[index]])

            
#   FUNCTION distance_
# Applies the distance_transform_edt function from scipy.ndimage.morphology
#   INPUTS:
# filename: name of hdf5 file containing data
# data_folder (default: 'datasets'): folder searched for image inputs
# selection (default: None): determines the name of folders or files to be used.
# criteria (default: None): determines what category selection refers to
#   OUTPUTS
# null

def distance_(filename, data_folder = 'datasets', selection = None, criteria = None):
    in_path_list = pt.path_inputs(filename, data_folder, selection, criteria)
    out_folder_locations = pt.find_output_folder_location(filename, 'distance', in_path_list)
    with h5py.File(filename, "a") as f:
        for index in range(len(in_path_list)):
            path = in_path_list[index]
            raw_data = np.copy(f[path])
            #indices = np.zeros(((np.ndim(~raw_data),) + raw_data.shape), dtype=np.int32)
            #final_distance = distance_transform_edt(~raw_data, return_indices=True,
                     #indices=indices)
            final_distance = distance_transform_edt(~raw_data)
            data = pt.write_output_f(f, final_distance, out_folder_locations[index],
                                     in_path_list[index], 'FeatureDistance')
            propagate_scale_attrs(data, f[in_path_list[index]])
            
            
#   FUNCTION directional_skeletonize_
# 'skeletonizes' a binary image either vertically or horizontally. This is done by finding the
# contours of each shape. The centre of each of these contours are then taken and extended either
# vertically or horizontally to the edge of each shape. If the edge of this shape is within 10
# pixels of the edge of the image, the line is further extended to the end of the image. Extra lines
# can also be removed via the false_positives variable
#   INPUTS:
# filename: name of hdf5 file containing data
# data_folder (default: 'datasets'): folder searched for image inputs
# selection (default: None): determines the name of folders or files to be used.
# criteria (default: None): determines what category selection refers to
# direction (default: 'Vert'): Direction of the a skeletonization process:
#     'Vert': Draws vertical lines
#     'Horz': Draws horizontal lines
# false_positives (default: None): a list of ints that defines which lines to be ignored. Each line
#     is described by an int, starting from number 0, which is the left- or up-most line.
# max_edge (default: 10): the distance a line will stretch to read the edge or another line
# overwrite (default: False): if set to True, if this function was the last process run, the last
#     run will be overwritten and replaced with this. To be used sparingly, and only if function
#     parameters must be guessed and checked
#   OUTPUTS
# null
            
def directional_skeletonize_(filename, data_folder = 'datasets', selection = None, criteria = None,
                             direction = 'Vert', false_positives = None, max_edge = 10,
                             overwrite = False):
    in_path_list = pt.path_inputs(filename, data_folder, selection, criteria)
    out_folder_locations = pt.find_output_folder_location(filename, 'directional_skeletonize',
                                                          in_path_list, overwrite)
    if (direction != 'Vert') and (direction != 'Horz'):
        print('direction should be set to either \'Vert\' or \'Horz\'')
        return
    if type(false_positives) != list:
        false_positives = [false_positives]
    with h5py.File(filename, "a") as f:
        for index in range(len(in_path_list)):
            path = in_path_list[index]
            domain_guess = np.copy(f[path]).astype(np.uint8)
            
            # find contours in the binary image
            image, contours, hierarchy = cv2.findContours(domain_guess,
                                                          cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            x_centres = []
            y_centres = []
            bad_domains = np.zeros_like(domain_guess)
            good_domains = np.zeros_like(domain_guess)
            for contour in contours:
                # calculate moments for each contour
                moment = cv2.moments(contour)
                
                # calculate x,y coordinate of center
                if moment['m00'] != 0:
                    x_centres.append(int(moment["m10"] / moment["m00"]))
                    y_centres.append(int(moment["m01"] / moment["m00"]))
                
            if direction == 'Vert':
                x_centres = sorted(x_centres)
                for i in range(len(x_centres)):
                    x = x_centres[i]
                    whole_line = domain_guess[:, x]
                    zero_count = 0
                    for y in range(len(whole_line)):
                        if whole_line[y]==0:
                            zero_count = zero_count + 1
                        else:
                            if zero_count <= max_edge:
                                whole_line[y-zero_count:y] = 1
                            zero_count = 0
                    if (zero_count != 0) and (zero_count <= max_edge):
                        whole_line[-zero_count:] = 1
                    if i in false_positives:
                        bad_domains[:,x] = whole_line 
                    else:
                        good_domains[:,x] = whole_line
            elif direction == 'Horz':
                y_centres = sorted(y_centres)
                for i in range(len(y_centres)):
                    y = y_centres[i]
                    whole_line = domain_guess[y]
                    zero_count = 0
                    for x in range(len(whole_line)):
                        if whole_line[x]==0:
                            zero_count = zero_count + 1
                        else:
                            if zero_count <= max_edge:
                                whole_line[x-zero_count:x] = 1
                            zero_count = 0
                    if (zero_count != 0) and (zero_count <= max_edge):
                        whole_line[-zero_count:] = 1
                    if i in false_positives:
                        bad_domains[y] = whole_line 
                    else:
                        good_domains[y] = whole_line
                            
            all_domains = bad_domains+good_domains
            data = pt.write_output_f(f, all_domains, out_folder_locations[index],
                                     in_path_list[index], 'AllLines')
            data = pt.write_output_f(f, good_domains, out_folder_locations[index],
                                     in_path_list[index], 'FilteredLines')
            if false_positives[0] is None:
                data.attrs['deleted a-domains'] = 'None'
            else:
                data.attrs['deleted a-domains'] = false_positives
                

#   FUNCTION final_a_domains_
# Creates a folder with all a-domain data from the directionally skeletonized binary images (both
# horz and vert). This includes the final cleaning steps, where both vertical and horizontal lines
# are overlayed and compared; if one line overlaps (or approaches) a perpendicular and ends at a
# distance less than that described by closing_distance, the extra line is cut (or the approaching
# line extended) such that the line terminates where it would coincide with the perpendicular. The
# final dataset folder contains images of both the horizontal and vertical a-domains, a composite
# image made from both the horizontal and vertical domains, and separate lists of both the
# horizontal and vertical domains that contain coordinates for a start and end of each domain
#   INPUTS:
# filename: name of hdf5 file containing data
# data_folder (default: 'datasets'): folder searched for vertical lines
# selection (default: 'FilteredLines'): determines the name of folders or files to be used.
# criteria (default: 'channel'): determines what category selection refers to
# hz_data_folder (default: 'datasets'): folder searched for horizontal lines
# hz_selection (default: 'FilteredLines'): determines the name of folders or files to be used.
# hz_criteria (default: 'channel'): determines what category selection refers to
# closing distance (default: 50): extra distance a line is extended to (or cut by) if it approaches
#     a perpendicular
#   OUTPUTS
# null
                
def final_a_domains_(filename, data_folder = 'datasets',
                     selection = 'FilteredLines', criteria = 'channel',
                     hz_data_folder = 'datasets',
                     hz_selection = 'FilteredLines', hz_criteria = 'channel',
                     closing_distance = 50):
    in_path_list = pt.path_inputs(filename, data_folder, selection, criteria)
    hz_path_list = pt.path_inputs(filename, hz_data_folder, hz_selection, hz_criteria)
    out_folder_locations = pt.find_output_folder_location(filename, 'final_a_domains', in_path_list)
    with h5py.File(filename, "a") as f:
        for index in range(len(in_path_list)):
            orig_vert = np.copy(f[in_path_list[index]])
            orig_horz = np.copy(f[hz_path_list[index]])
            
            new_vert = np.copy(orig_vert)
            #Lines defined by x1, y1, x2, y2
            vert_list = []
            for x in range(np.shape(new_vert)[1]):
                if np.sum(orig_vert[:,x]) != 0:
                    if np.sum(orig_vert[:,x]) == np.shape(new_vert)[0]:
                        vert_list.append([x, 0, x, np.shape(new_vert)[0]-1])
                    elif np.sum(orig_horz[:,x]) != 0:
                        domains_list = np.where(orig_horz[:,x]==1)[0]
                        for domain in domains_list:
                            min_index = np.max([0, domain-closing_distance])
                            max_index = np.min([domain+1+closing_distance, len(orig_horz[:,x])])
                            vert_top_segment = new_vert[min_index:domain+1, x]
                            if (np.sum(vert_top_segment) != 0) and (np.sum(vert_top_segment) !=
                                                                    np.shape(vert_top_segment)[0]):
                                if vert_top_segment[0] == 1:
                                    new_vert[min_index:domain+1,x]=np.zeros_like(vert_top_segment)+1
                                else:
                                    new_vert[min_index:domain+1,x]=np.zeros_like(vert_top_segment)
                            vert_bot_segment = new_vert[domain:max_index, x]
                            if (np.sum(vert_bot_segment) != 0) and (np.sum(vert_bot_segment) !=
                                                                    np.shape(vert_bot_segment)[0]):
                                if vert_bot_segment[-1] == 1:
                                    new_vert[domain:max_index, x]=np.zeros_like(vert_bot_segment)+1
                                else:
                                    new_vert[domain:max_index, x]=np.zeros_like(vert_bot_segment)
                        line_found = False
                        for y in range(np.shape(new_vert)[0]):
                            if (new_vert[y,x] == 1) and (not line_found):
                                line_found = True
                                y1 = y
                            if (new_vert[y,x] == 0) and line_found:
                                vert_list.append([x, y1, x, y-1])
                                line_found = False
                        if line_found == True:
                            vert_list.append([x, y1, x, np.shape(new_vert)[0]-1])
                                        
            new_horz = np.copy(orig_horz)
            horz_list = []
            for y in range(np.shape(new_horz)[0]):
                if np.sum(orig_horz[y,:]) != 0:
                    if np.sum(orig_horz[y,:]) == np.shape(new_horz)[1]:
                        horz_list.append([0, y, np.shape(new_horz)[1]-1, y])
                    elif np.sum(orig_vert[y,:]) != 0:
                        domains_list = np.where(orig_vert[y,:]==1)[0]
                        for domain in domains_list:
                            min_index = np.max([0, domain-closing_distance])
                            max_index = np.min([domain+1+closing_distance, len(orig_vert[y,:])])
                            horz_lft_segment = new_horz[y, min_index:domain+1]
                            if (np.sum(horz_lft_segment) != 0) and (np.sum(horz_lft_segment) != 
                                                                    np.shape(horz_lft_segment)[0]):
                                if horz_lft_segment[0] == 1:
                                    new_horz[y,min_index:domain+1]=np.zeros_like(horz_lft_segment)+1
                                else:
                                    new_horz[y,min_index:domain+1]=np.zeros_like(horz_lft_segment)
                            horz_rgt_segment = new_horz[y, domain:max_index]
                            if (np.sum(horz_rgt_segment) != 0) and (np.sum(horz_rgt_segment) != 
                                                                    np.shape(horz_rgt_segment)[0]):
                                if horz_rgt_segment[-1] == 1:
                                    new_horz[y, domain:max_index]=np.zeros_like(horz_rgt_segment)+1
                                else:
                                    new_horz[y, domain:max_index]=np.zeros_like(horz_rgt_segment)
                        line_found = False
                        for x in range(np.shape(new_horz)[1]):
                            if (new_horz[y,x] == 1) and (not line_found):
                                line_found = True
                                x1 = x
                            if (new_horz[y,x] == 0) and line_found:
                                horz_list.append([x1, y, x-1, y])
                                line_found = False
                        if line_found == True:
                            horz_list.append([x1, y, np.shape(new_horz)[1]-1, y])
                                
            new_all = np.maximum(new_vert, new_horz)
            
            data = pt.write_output_f(f, new_horz, out_folder_locations[index], in_path_list[index],
                                     'HorzADomains')
            data = pt.write_output_f(f, new_vert, out_folder_locations[index], in_path_list[index],
                                     'VertADomains')
            data = pt.write_output_f(f, new_all, out_folder_locations[index], in_path_list[index],
                                     'AllADomains')            
            data = pt.write_output_f(f, np.array(horz_list), out_folder_locations[index], 
                                     in_path_list[index], 'HorzADomainsList')
            data = pt.write_output_f(f, np.array(vert_list), out_folder_locations[index], 
                                     in_path_list[index], 'VertADomainsList')
            
            
#   FUNCTION switchmap_
# Generates a switchmap from binarised phase data. A switchmap is a 2D array, where the number
# at each coordinate corresponds to the 'time' it takes that coordinate to switch phase. If a
# coordinate did not switch, it is set to a NaN.
#   INPUTS:
# filename: name of hdf5 file containing data
# data_folder (default: 'process/01-phase_binarisation'): folder searched for binarised phases
# selection (default: None): determines the name of folders or files to be used.
# criteria (default: None): determines what category selection refers to
# method (default: 'total'): determines the method used to generate the switchmap:
#     'maximum': switching occurs at the final time the coordinate switches
#     'minimum': switching occurs at the first time the coordinate switches
#     'median': switching occurs at the median of all times the coordinate switches
#     'total': switching occurs at the number of total scans that the coordinate is not switched
# voltage_in_title (default: False): if set to True, the function will attempt to read the voltage
#     (in mV) from the dataset title. For this to work, the title must contain the voltage in the
#     form '_*mV', where * is the number that is read.
#   OUTPUTS
# null

def switchmap_(filename, data_folder = 'process/01-phase_binarisation', selection = None,
               criteria = None, method = 'total', voltage_in_title = False):
    in_path_list = pt.path_inputs(filename, data_folder, selection, criteria)
    out_folder_locations = pt.find_output_folder_location(filename, 'switchmap',
                                                          filename.split('.')[0])
    with h5py.File(filename, "a") as f:
        phase_list = []
        voltage_list = []
        for index in range(len(in_path_list)):
            path = in_path_list[index]
            curr_array = np.copy(f[path])
            curr_array = curr_array.astype(float)
            phase_list.append(curr_array)
            
            if voltage_in_title:
                mV = path.split('mV')[-2]
                mV = mV.split('_')[-1]
                voltage_list.append(int(mV))
        
        switchmap = np.zeros_like(phase_list[0].astype(float))
        for i in range(phase_list[0].shape[0]):
            for j in range(phase_list[0].shape[1]):
                switch_list = []
                for phase in phase_list:
                    switch_list.append(phase[i,j])
                if (switch_list[0]==switch_list[-1]):
                    switchmap[i,j] = np.nan
                else:
                    changes=wherechanged(switch_list)
                    if method=='maximum':
                        switch_scan=np.ceil(np.nanmax(changes))
                    elif method == 'minimum':
                        if len(changes) == changes[-1]:
                            switch_scan = changes[-1]
                        else:
                            scan = 0
                            while changes[scan] == scan+1:
                                scan = scan+1
                            switch_scan = changes[scan-1]
                    elif method=='median':
                        switch_scan=np.ceil(np.nanmedian(changes))
                    elif method=='total':
                        switch_scan=np.sum(switch_list)+1
                    else:
                        print('Error: Invalid method submitted')
                    switchmap[i,j] = switch_scan        
        data = pt.write_output_f(f, switchmap, out_folder_locations[0], in_path_list, 'Switchmap')
        propagate_scale_attrs(data, f[in_path_list[index]])
        if voltage_in_title:
            data.attrs['voltage (mV)'] = voltage_list
            
            
#   FUNCTION wherechanged
# Returns the indices where a list changes values
#   INPUTS:
# arr: the array or list that may contain changes
#   OUTPUTS
# where: a list of indices that changes occur
            
def wherechanged(arr):
    where=[]
    for i in range(len(arr)-1):
        diff=arr[i+1]-arr[i]
        if diff!=0:
            where.append(i+1)
    return where


#   FUNCTION switch_type_
# Generates data regarding the type of switch. This data is placed into two folders. In switch_type,
# a folder is made for each scan. This folder contains an array, where the number defines the type
# of switch that had occurred at that coordinate by that point (no distinction is made in the
# recency of switches). NaN = no switch; 1 = nucleation; 2 = motion; 3 = merging; 4 = errors. These
# are defined by the amount of neighbours for a switch (nucleation has 0, motion has 1, merging has
# 2, errors have another value). The second folder, switch_type_general, holds information common to
# the entire switchmap. In the 'Centres' subfolder, two 2D arrays are given. NucleationCentres has 1 
# at the centre of a nucleation, and 0 elsewhere, while ClosureCentres has 1 at the centre of a
# closure (the opposite of nucleation; where no more motion or merging can continue off of it), and
# 0 elsewhere (note that as closure can be either motion or merging, switch_type does not contain
# any information on closure). In the "JumpTypes" subfolder, 6 arrays are stored, which contain
# information on the type of switch. These are the four types of switching used in switch_type
# (Nucleation; Motion; Merging; and Errors), as well as one for all switching (TotalJumps) and one
# for closure (Closure), which has redundancy with Motion and Merging. In these arrays, each row
# signifies scan. Each entry shows the size of a switch at that scan. As the length of each row
# is thus arbitirary, NaNs are made to fill each row to ensure constant length.
#   INPUTS:
# filename: name of hdf5 file containing data
# data_folder (default: 'process/01-switchmap'): folder searched for switchmap
# selection (default: None): determines the name of folders or files to be used.
# criteria (default: None): determines what category selection refers to
#   OUTPUTS
# null

def switch_type_(filename, data_folder = 'process/01-switchmap', selection = None, criteria = None):
    in_path_list = pt.path_inputs(filename, data_folder, selection, criteria)
    out_folder_locations = pt.find_output_folder_location(filename, 'switch_type', '')
    with h5py.File(filename, "a") as f:
        switchmap = np.copy(f[in_path_list[0]])
        
        if 'voltage (mv)' in f[in_path_list[0]].attrs:
            voltage_array = f[in_path_list[0]].attrs['voltage (mV)']
        else:
            voltage_array = None
        total_scans = 0
        for attr in f[in_path_list[0]].attrs:
            if 'source' in attr:
                total_scans = total_scans+1
        
        totalmap=np.zeros_like(switchmap.astype(float))
        totalmap[:]=np.nan
        
        alljumps_tot=[]
        alljumps_nucl=[]
        alljumps_mot=[]
        alljumps_merg=[]
        alljumps_error=[]
        alljumps_closure=[]
        
        nucl_centres=np.zeros_like(switchmap.astype(bool))
        closure_centres=np.zeros_like(switchmap.astype(bool))

        for i in range(total_scans):
            alljumps_tot_1img=[]
            alljumps_nucl_1img=[]
            alljumps_mot_1img=[]
            alljumps_merg_1img=[]
            alljumps_error_1img=[]
            alljumps_closure_1img=[]
            
            # Extract when switching occured
            prev_scan=(switchmap<=i).astype(int)
            curr_scan=(switchmap<=i+1).astype(int)
            switched_regions = curr_scan-prev_scan

            # Label areas where switching occured
            structuring_element=[[1,1,1],[1,1,1],[1,1,1]]
            labeled_switched_regions, total_features=label(switched_regions, structuring_element)

            for m in range(total_features):
                feature_region = (labeled_switched_regions == m+1)
                
                #Get positions of all points of each label
                feature_positions=np.nonzero(feature_region)
                label_surface=np.sum(feature_region)

                #define box around each labelled pixel        
                box=np.zeros_like(switchmap)
                box = binary_dilation(feature_region, iterations = 1)
                box_border = box^(feature_region)

                if np.array_equal(np.logical_and(box_border, prev_scan),box_border):
                    closure = True
                else:
                    closure = False

                #Get switched areas up to the previous scan, that are in the box
                boxed_pha_bin=np.zeros_like(switchmap)
                boxed_pha_bin=(prev_scan+np.isnan(switchmap))*box_border   #Possibly remove negative

                #Label the switched areas in the box
                labeled_boxed_pha_bin, num_connectors=label(boxed_pha_bin, structuring_element)

                #Convert the labeled array into a 1d-array of (unique) label values whose
                #length gives the number of domains connecting the newly switched area
                num_pix=np.shape(switchmap)[0]*np.shape(switchmap)[1]
                boxed_labels=np.reshape(labeled_boxed_pha_bin, (1,num_pix))
                boxed_labels=np.unique(boxed_labels[boxed_labels!=0])
                number_of_connecting_doms=len(boxed_labels)

                #Define event type and append all points of that event to the totalmap
                #nucleation
                alljumps_tot_1img.append(label_surface)            
                if number_of_connecting_doms==0:
                    eventtype=2
                    alljumps_nucl_1img.append(label_surface)
                    event_centre = skeletonize(feature_region, method='lee').astype(bool)
                    nucl_centres = nucl_centres+event_centre
                    for j in range(len(feature_positions[0])):
                        x=feature_positions[0][j]
                        y=feature_positions[1][j]
                        totalmap[x,y]=2.

                #motion
                elif number_of_connecting_doms==1:
                    eventtype=1
                    alljumps_mot_1img.append(label_surface)
                    for j in range(len(feature_positions[0])):
                        x=feature_positions[0][j]
                        y=feature_positions[1][j]
                        totalmap[x,y]=1.
                #merging
                elif number_of_connecting_doms>1:
                    eventtype=3
                    alljumps_merg_1img.append(label_surface)
                    for j in range(len(feature_positions[0])):
                        x=feature_positions[0][j]
                        y=feature_positions[1][j]
                        totalmap[x,y]=3.
                else:
                    eventtype=4
                    alljumps_error_1img.append(label_surface)
                    for j in range(len(feature_positions[0])):
                        x=feature_positions[0][j]
                        y=feature_positions[1][j]
                        totalmap[x,y]=4.
                #closure
                if closure==True:
                    alljumps_closure_1img.append(label_surface)
                    event_centre = skeletonize(feature_region, method='lee').astype(bool)
                    closure_centres = closure_centres+event_centre

            alljumps_tot.append(alljumps_tot_1img)
            alljumps_nucl.append(alljumps_nucl_1img)
            alljumps_mot.append(alljumps_mot_1img)
            alljumps_merg.append(alljumps_merg_1img)
            alljumps_error.append(alljumps_error_1img)
            alljumps_closure.append(alljumps_closure_1img)
                   
            if voltage_array is not None:
                name = str(voltage_array[i]).zfill(4)+'_mV'
            else:
                name = 'Scan_'+str(i).zfill(3)
            current_folder_location = out_folder_locations[0]+name
            data = pt.write_output_f(f, totalmap, current_folder_location, in_path_list,
                                     'Switchmap')
            propagate_scale_attrs(data, f[in_path_list[0]])
        
        gen_loc = pt.find_output_folder_location(filename, 'switch_type_general', 'Centres')[0]
        data = pt.write_output_f(f, nucl_centres, gen_loc, in_path_list, 'NucleationCentres')
        data = pt.write_output_f(f, closure_centres, gen_loc, in_path_list, 'ClosureCentres')
        
        gen_loc = pt.find_output_folder_location(filename, 'switch_type_general', 'JumpTypes',
                                                 True)[0]
        data = pt.write_output_f(f, fill_blanks(alljumps_tot), gen_loc, in_path_list, 'TotalJumps')
        data = pt.write_output_f(f, fill_blanks(alljumps_nucl), gen_loc, in_path_list, 'Nucleation')
        data = pt.write_output_f(f, fill_blanks(alljumps_mot), gen_loc, in_path_list, 'Motion')
        data = pt.write_output_f(f, fill_blanks(alljumps_merg), gen_loc, in_path_list, 'Merging')
        data = pt.write_output_f(f, fill_blanks(alljumps_error), gen_loc, in_path_list, 'Errors')
        data = pt.write_output_f(f, fill_blanks(alljumps_closure), gen_loc, in_path_list, 'Closure')

        
#   FUNCTION switch_type_
# Takes a list of lists, and extends each individual list such that each list is the same size. This
# is done by introducing NaNs. An list that is, for example, [[1,2,3],[],[1]], would then become
# [[1,2,3],[np.nan, np.nan, np.nan],[1, np.nan, np.nan]]
#   INPUTS:
# list_of_lists: list of lists to be altered so each component list is the same length
#   OUTPUTS
# list_of_lists: the new list of lists, where each list is extended to the same length
        
def fill_blanks (list_of_lists):
    longest_list_length = 0
    for one_list in list_of_lists:
        longest_list_length = max(longest_list_length, len(one_list))
    for one_list in list_of_lists:
        extra_nans = longest_list_length - len(one_list)
        one_list.extend([np.nan] * extra_nans)
    return list_of_lists


#   FUNCTION interpolated_features_
# Interpolates features on a switchmap. This is done by first isolating key points: these are either
# the edges of the switchmap themselves, or the the centre of closure or nucleation. Using these key
# points, the image is interpolated linearly. Regions that cannot be interpolated, such as regions
# that did not switch, or corners, are set to NaN.
#   INPUTS:
# filename: name of hdf5 file containing data
# data_folder (default: 'process/01-switchmap'): folder searched for switchmap
# selection (default: None): determines the name of folders or files to be used.
# criteria (default: None): determines what category selection refers to
# centre_data_folder (default: 'process/02-switch_type_general'): folder searched for nucleation and
#     closure centres
# centre_selection (default: None): determines the name of folders or files to be used.
# centre_criteria (default: None): determines what category selection refers to
#   OUTPUTS
# null

def interpolated_features_(filename, data_folder = 'process/01-switchmap', selection = None,
                           criteria = None, centre_data_folder = 'process/02-switch_type_general',
                           centre_selection = 'Centres', centre_criteria = 'Sample'):
    in_path_list = pt.path_inputs(filename, data_folder, selection, criteria)
    out_folder_locations = pt.find_output_folder_location(filename, 'interpolated_features',
                                                          in_path_list)
    centre_path_list = pt.path_inputs(filename, centre_data_folder, centre_selection,
                                      centre_criteria)
    with h5py.File(filename, "a") as f:
        for path in centre_path_list:
            if ('Nucl' or 'nucl') in path:
                nucl_centres = np.copy(f[path])
            if ('Clos' or 'clos') in path:
                clos_centres = np.copy(f[path])
        for index in range(len(in_path_list)):
            path = in_path_list[index]
            switchmap = np.copy(f[path])
            isolines = find_isolines(switchmap, nucl_centres, clos_centres)
            isoline_y = []
            isoline_x = []
            isoline_z = []
            for i in range(np.shape(isolines)[0]):
                for j in range(np.shape(isolines)[1]):
                    if isolines[i,j]!=0:
                        isoline_x.append(j)
                        isoline_y.append(i)
                        isoline_z.append(isolines[i,j])

            grid_x, grid_y = np.mgrid[0:np.shape(isolines)[0]:1, 0:np.shape(isolines)[1]:1]
            interpolation = interpolate.griddata(np.array([isoline_y, isoline_x]).T, isoline_z,
                                                 (grid_x, grid_y), method='linear',
                                                 fill_value = np.nan)
            data = pt.write_output_f(f, interpolation, out_folder_locations[index],
                                     [in_path_list, centre_path_list])
            propagate_scale_attrs(data, f[in_path_list[index]])
            
            
#   FUNCTION find_isolines
# Finds key features on a switchmap (or similar structures) and corresponding nucleation and closure
# centres. Three features are considered: the edges of each switch on the switchmap ('cliffs') are 
# set to the value of the switchmap; the nucleation centres are set to 0.5 less than the value at
# the same point on the switchmap; and the closure centres are set to 0.5 more.
#   INPUTS:
# switchmap: the switchmap used as the base for key features
# nucl_centres: points where nucleation occurs
# clos_centres: points where closure occurs
#   OUTPUTS
# isolines: the key features on the switchmap, including the isolines themselves, and the nucleation
#     and closure centres.
            
def find_isolines(switchmap, nucl_centres, clos_centres):
    isolines = np.zeros_like(switchmap)
    for i in range(np.shape(switchmap)[0]):
        for j in range(np.shape(switchmap)[1]):
            peak = False
            if np.isnan(switchmap[i,j]):
                isolines[i,j] = np.nan
            if i != 0:
                if np.isnan(switchmap[i-1,j]) or (switchmap[i,j]>switchmap[i-1,j]):
                    isolines[i,j]=switchmap[i,j]
                    peak = True
                elif switchmap[i,j]<switchmap[i-1,j]-1 and peak == False:
                    isolines[i,j]=switchmap[i,j]+0.5
            if i != np.shape(switchmap)[0]-1:
                if np.isnan(switchmap[i+1,j]) or (switchmap[i,j]>switchmap[i+1,j]):
                    isolines[i,j]=switchmap[i,j]
                    peak = True
                elif switchmap[i,j]<switchmap[i+1,j]-1 and peak == False:
                    isolines[i,j]=switchmap[i,j]+0.5
            if j != 0:
                if np.isnan(switchmap[i,j-1]) or (switchmap[i,j]>switchmap[i,j-1]):
                    isolines[i,j]=switchmap[i,j]
                    peak = True
                elif switchmap[i,j]<switchmap[i,j-1]-1 and peak == False:
                    isolines[i,j]=switchmap[i,j]+0.5
            if j != np.shape(switchmap)[1]-1:
                if np.isnan(switchmap[i,j+1]) or (switchmap[i,j]>switchmap[i,j+1]):
                    isolines[i,j]=switchmap[i,j]
                    peak = True
                elif switchmap[i,j]<switchmap[i,j+1]-1 and peak == False:
                    isolines[i,j]=switchmap[i,j]+0.5
    clos_location = np.where(clos_centres)
    for centre_number in range(np.shape(clos_location)[1]):
        i = clos_location[0][centre_number]
        j = clos_location[1][centre_number]
        isolines[i,j] = switchmap[i,j]+0.5
    nucl_location = np.where(nucl_centres)
    for centre_number in range(np.shape(nucl_location)[1]):
        i = nucl_location[0][centre_number]
        j = nucl_location[1][centre_number]
        isolines[i,j] = switchmap[i,j]-0.5
    return isolines


#   FUNCTION differentiate_
# Differentiates an image. Creates files corresponding to the magnitude, of the derivative and
# optionally, derivatives along both axis separately.
#   INPUTS:
# filename: name of hdf5 file containing data
# data_folder (default: 'datasets'): folder searched for image data
# selection (default: None): determines the name of folders or files to be used.
# criteria (default: None): determines what category selection refers to
# all_directions (default: True): If set to false, only the derivative magnitude is stored.
#   OUTPUTS
# null

def differentiate_(filename, data_folder = 'datasets', selection = None, criteria = None,
                   all_directions = True):
    in_path_list = pt.path_inputs(filename, data_folder, selection, criteria)
    out_folder_locations = pt.find_output_folder_location(filename, 'differentiate', in_path_list)
    with h5py.File(filename, "a") as f:
        for index in range(len(in_path_list)):
            path = in_path_list[index]
            raw_data = np.copy(f[path])
            deriv = np.abs(np.gradient(raw_data))
            deriv = np.sqrt(deriv[0]**2+deriv[1]**2)
            data = pt.write_output_f(f, deriv, out_folder_locations[index], in_path_list,
                                     'AbsDerivative')
            propagate_scale_attrs(data, f[in_path_list[index]])
            data.attrs['dimension'] = 'Abs'
            
            if all_directions:
                deriv = np.gradient(raw_data)
                for i in range(len(np.shape(raw_data))):
                    data = pt.write_output_f(f, deriv[i], out_folder_locations[index], in_path_list,
                                             'Derivative'+str(i))
                    propagate_scale_attrs(data, f[in_path_list[index]])
                    data.attrs['dimension'] = i
                    
