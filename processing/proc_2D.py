import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
from . import proc_tools as pt
import cv2
import os

#FUNCTION save_image 
#INPUTS:
## data: A 2-D array which will be converted into a png image.
## scalebar (default: False): Add a scalebar to the image, requires three attributes : 
##                                                 shape, which define the pixel size of the image
##                                                 size, which gives the phyiscal dimension of the image
##                                                 unit, which give the physical unit of size
## size (default: None : Dimension of the saved image. If none, the image is set to have one pixel per data point at 100 dpi
## labelsize (default: 25) : Size of the text in pxs
## std_range (default: 3) : Range around the mean for the colorscale, alternatively the value can be "full", to take the full range.
## saving_path (default: '') : The path to the folder where to save the image
## verbose (default: False) : if True, print a line each time a image is saved.
## Output : one png images
## TO DO: Allow for no border at all

def save_image(data, 
               image_name='image', 
               colorm='inferno',
               scalebar=False,
               physical_size = (0, 'unit'),
               colorbar = True, 
               size=None, 
               labelsize=16, 
               std_range=3, 
               saving_path='', 
               verbose=False): 
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
                phys_size = physical_size[0]
                px = np.shape(data)[0]
                scalebar = ScaleBar(phys_size/px, physical_size[1], location='lower right', font_properties={'size':labelsize})
                fig.add_artist(scalebar)
            except:
                print("Error in the creation of the scalebar, please check that the attributes size and shape are correctly define for each datas channels.")
        fig.savefig(saving_path+str(image_name)+'.png')
        if verbose:
            print(filename.split('.')[0]+'_'+str(image_name)+'.png saved.')
        plt.close()
    return

#FUNCTION distortion_params_
## determine cumulative translation matrices for distortion correction.
#INPUTS:
## filename: name of hdf5 file containing data
## data_folder: folder searched for inputs. eg. 'datasets', or 'process/negative'
## selection: determines the name of folders or files to be used. Can be None (selects all), a string, or a list of strings. Default allows for correction based on topography in data from Asylum AFM.
## criteria (default: 'channel'): determines the type of data selected. Set to 'process', 'sample' or 'channel'. Default allows for correction based on topography in data from Asylum AFM.
## speed: determines speed and accuracy of function. An integer between 1 and 3, a higher number if faster (but needs less distortion to work)
#OUTPUTS:
## null

def distortion_params_(filename, data_folder='datasets', selection = 'HeightRetrace', criteria = 'channel', speed = 2):
    in_path_list = pt.path_inputs(filename, data_folder, selection, criteria)
    out_folder_locations = pt.find_output_folder_location(filename, 'distortion_params', in_path_list)
    tform21 = np.eye(2,3,dtype=np.float32)
    cumulative_tform21 = np.eye(2,3,dtype=np.float32)
    with h5py.File(filename, "a") as f:
        recent_offsets=[]
        for i in range(len(in_path_list)):
            if i == 0:
                pass
            else:
                img1 = img2cv((f[in_path_list[i-1]]))
                img2 = img2cv((f[in_path_list[i]]))
                
                # try estimate offset change from attribs of img1 and img2
                try:
                    offset2 = (f[in_path_list[i]]).attrs['offset']
                    offset1 = (f[in_path_list[i-1]]).attrs['offset']
                    scan_size = (f[in_path_list[i]]).attrs['size']
                    shape = (f[in_path_list[i]]).attrs['shape']
                    offset_px = m2px(offset2-offset1, shape, scan_size)
                except:
                    offset_px = [0,0]
                
                if speed != 1 and speed != 2 and speed != 3 and speed != 4:
                    print('Error: Speed should be an integer between 1 (slowest) and 4 (fastest). Speed now set to level 2.')
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
                    offset_guess = offset_px + recent_offsets[2]/2 + recent_offsets[1]/3 + recent_offsets[0]/6
                    if speed == 1:
                        warp_check_range = 8
                    elif speed == 2:
                        warp_check_range = 6
                    elif speed == 3:
                        warp_check_range = 4
                    elif speed == 4:
                        warp_check_range = 2

                tform21 = generate_transform_xy(img1, img2, tform21, offset_guess, warp_check_range)
                cumulative_tform21[0,2]=cumulative_tform21[0,2]+tform21[0,2]
                cumulative_tform21[1,2]=cumulative_tform21[1,2]+tform21[1,2]
                print('Scan '+str(i)+' Complete. Cumulative Transform Matrix:')
                print(cumulative_tform21)
                
                recent_offsets.append([tform21[0,2], tform21[1,2]]-offset_px)
                if len(recent_offsets)>3:
                    recent_offsets = recent_offsets[1:]
            pt.write_output_f(f, cumulative_tform21, out_folder_locations[i], in_path_list[i])
        

#FUNCTION m2px
## Converts length in metres to a length in pixels
#INPUTS:
## m: length in metres to be converted
## points: number of lines or points per row
## scan_size: total length of scan
#OUTPUTS:
## converted length in pixels
        
def m2px (m, points, scan_size):
    px = m*points/scan_size
    return px
        
        
#FUNCTION img2cv
## Converts img (numpy array, or hdf5 dataset) into cv2
#INPUTS:
## img1: currently used image
## sigma_cutoff: ???
#OUTPUTS:
## converted image into cv2 valid format

def img2cv(img1, sigma_cutoff=10):
    #img1 = np.diff(img1)
    img1 = img1-np.min(img1)
    img1 = img1/np.max(img1)
    tmp1 = sigma_cutoff*np.std(img1)
    img1[img1>tmp1] = tmp1
    return img1


#FUNCTION generate_transform_array
## Determines transformation matrixes in x and y
## Slow and primitive: needs updating
#INPUTS:
## img: currently used image (in cv2 format) to find transformation array of
## img_orig: image (in cv2 format) to find transformation array is based off of
## tfinit: base array passed into function
## offset_guess: Initial estimate of distortion, in pixels
## warp_check_range: total distance (in pixels) examined. Number of iterations = (warp_check_range+1)**2
#OUTPUTS:
## Transformation images used to convert img_orig into img

def generate_transform_xy(img, img_orig, tfinit=None, offset_guess = [0,0], warp_check_range=10):
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

    criteria = (term_flags, number_of_iterations, termination_eps)

    diff = np.Inf
    for i in range(-warp_check_range//2,(warp_check_range//2)+1):
        for j in range(-warp_check_range//2,(warp_check_range//2)+1):
            warp_matrix[0,2] = 2*i + offset_guess[0]
            warp_matrix[1,2] = 2*j + offset_guess[1]
            try:
                (cc, tform21) = cv2.findTransformECC(img_orig, img, warp_matrix, warp_mode, criteria)
                img_test = cv2.warpAffine(img, tform21, (512,512), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
                currDiff = np.sum(np.square(img_test[150:-150, 150:-150]-img_orig[150:-150, 150:-150]))
                if currDiff < diff:
                    diff = currDiff
                    offset1 = tform21[0,2]
                    offset2 = tform21[1,2]
            except:
                pass
            warp_matrix[0,2] = offset1
            warp_matrix[1,2] = offset2
    return warp_matrix

#FUNCTION distortion_correction_
## applies distortion correction params to an image
#INPUTS:
## filename: name of hdf5 file containing data
## data_folder: folder searched for image inputs. eg. 'datasets', or 'process/negative'
## selection: determines the name of folders or files to be used. Can be None (selects all), a string, or a list of strings. Default allows for correction based on topography in data from Asylum AFM.
## selection_depth: determines what level at which to look at a selection. Default allows for correction based on topography in data from Asylum AFM.
## dm_data_folder: folder searched for distortion matrix inputs. eg. 'datasets', or 'process/negative'
## dm_selection: determines the name of folders or files to be used. Can be None (selects all), a string, or a list of strings. Default allows for extraction if distortion parameters were the first processing funciton applied.
## dm_selection_depth: determines what level at which to look at a selection. Default allows for extraction if distortion parameters were the first processing funciton applied.
#OUTPUTS:
## null

def distortion_correction_(filename, data_folder='datasets', selection=None, criteria=None, dm_data_folder = 'process/1-distortion_params', dm_selection=None, dm_criteria=None, cropping = True):
    #dm_path_lists = pt.initialise_process(filename, None, data_folder=dm_data_folder, selection=dm_selection, selection_depth=dm_selection_depth)
    
    dm_path_list = pt.path_inputs(filename, dm_data_folder, dm_selection, dm_criteria)
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

    #path_lists = pt.initialise_process(filename, 'distortion_correction', data_folder=data_folder, selection=selection, selection_depth=selection_depth)
    in_path_list = pt.path_inputs(filename, data_folder, selection, criteria)
    out_folder_locations = pt.find_output_folder_location(filename, 'distortion_correction', in_path_list)
    if len(in_path_list)%len(dm_path_list):
        print('Error: Images to be corrected are not a multiple of the amount of distortion matrices')
        return

    number_of_images_for_each_matrix = len(in_path_list)//len(dm_path_list)
    with h5py.File(filename, "a") as f:
        j = -1
        for i in range(len(in_path_list)):
            if i%number_of_images_for_each_matrix == 0:
                j = j+1
            orig_image = f[in_path_list[i]]
            if cropping == True:
                final_image = array_cropped(orig_image, xoffsets[j], yoffsets[j], offset_caps)
            else:
                final_image = array_expanded(orig_image, xoffsets[j], yoffsets[j], offset_caps)
            pt.write_output_f(f, final_image, out_folder_locations[i], [in_path_list[i], dm_path_list[j]])
            
            #pt.generic_write(f, in_path_list[i], final_image, 'source (distortion params)', dm_path_list[j], 'distortion_params', distortion_matrices[j])

            
#FUNCTION array_cropped
## crops a numpy_array given the offsets of the array, and the minimum and maximum offsets of a set, to include only valid data shared by all arrays
#INPUTS:
## array: the array to be cropped
## xoffset: the xoffset of the array
## yoffset: the yoffset of the array
## offset_caps: a list of four entries. In order, these entries are the xoffset maximum, xoffset minimum, yoffset maximum, and yoffset minimum fo all arrays
#OUTPUTS:
## the cropped array

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


#FUNCTION array_expanded
## expands a numpy_array given the offsets of the array, and the minimum and maximum offsets of a set, to include all points of each array. Empty data is set to be NaN
#INPUTS:
## array: the array to be expanded
## xoffset: the xoffset of the array
## yoffset: the yoffset of the array
## offset_caps: a list of four entries. In order, these entries are the xoffset maximum, xoffset minimum, yoffset maximum, and yoffset minimum fo all arrays
#OUTPUTS:
## the expanded

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