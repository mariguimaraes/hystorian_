import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
from . import proc_tools as pt
import cv2
import os

#FUNCTION save_image 
#INPUTS:
## filename : The hdf5 file containing the data you want to convert into png
## data_folder (default : 'datasets'): Choose the folder that needs to be saved, by default the raw datas are used
## selection (default : None): determines the name of folders or files to be used. Can be None (selects all), a string, or a list of strings
## selection_depth (default: 0): determines what level at which to look at a selection.
## scalebar (default: False): Add a scalebar to the image, requires three attributes : 
##                                                 shape, which define the pixel size of the image
##                                                 size, which gives the phyiscal dimension of the image
##                                                 unit, which give the physical unit of size
## size (default: None : Dimension of the saved image. If none, the image is set to have one pixel per data point at 100 dpi
## labelsize (default: 25) : Size of the text in pxs
## std_range (default: 3) : Range around the mean for the colorscale, alternatively the value can be "full", to take the full range.
## saving_path (default: '') : The path to the folder where to save the image
## verbose (default: False) : if True, print a line each time a image is saved.
## Output : N png images, where N is the number of datas channels in the hdf5 file.
## TO DO: Allow for autocalculation of size params


def save_image(filename,data_folder='datasets', selection=None, selection_depth=0, scalebar=False, colorbar = True, size=None, labelsize=25, std_range=3, saving_path='', verbose=False): 
    erase = False
    std_range = float(std_range)
    if filename.split('.')[-1] != 'hdf5':
        try:
            read_file.tohdf5(filename)
            filename = filename.split('.')[0] + '.hdf5'
            erase = True
        except:
            print("File extension is not hdf5 and it was not possible to convert it, please convert it before using this function")
            return
    path_list = pt.initialise_process(filename, None, data_folder=data_folder, selection=selection, selection_depth=selection_depth)
    with h5py.File(filename, "r") as f:
        for path in path_list:
            image_name = path.rsplit('/')[-1]
            if size is None:
                fig = plt.figure(frameon=False, figsize=(np.array(np.shape(f[path]))/100)[::-1], dpi=100)
            else:
                fig = plt.figure(figsize=size)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            plt.tick_params(labelsize=labelsize)
            #print(image_name)
            if 'Phase' in image_name:
                colorm = 'inferno'
                offsetdata = f[path] - np.nanmin(f[path])
                #print(np.min(offsetdata))
                v_min = 0
                v_max = 180
                pos = ax.imshow(offsetdata, vmin=v_min, vmax=v_max, cmap=colorm)
            if 'Amplitude' in image_name:
                colorm = 'binary_r'
                offsetdata = f[path] - np.nanmin(f[path])
                #print(np.min(offsetdata))
                mean_val = np.nanmean(offsetdata)
                std_val = np.nanstd(offsetdata)
                v_min = 0
                v_max = mean_val + std_range*std_val
                pos = ax.imshow(offsetdata, vmin=v_min, vmax=v_max, cmap=colorm)
            else:
                colorm = 'afmhot'
                offsetdata = f[path] - np.nanmin(f[path])
                #print(np.min(offsetdata))
                if std_range == "full":
                    pos = ax.imshow(offsetdata, cmap=colorm)
                else:
                    try:
                        mean_val = np.nanmean(offsetdata)
                        std_val = np.nanstd(offsetdata)
                        v_min = 0
                        v_max = mean_val + std_range*std_val
                        #print(v_max)
                        pos = ax.imshow(offsetdata, vmin=v_min, vmax=v_max, cmap=colorm)
                    except:
                        print("error in the min, max for the image, whole range is used.")
                        pos = ax.imshow(offsetdata, cmap=colorm)
            if colorbar == True:
                cbar = plt.colorbar(pos,fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=25) 
            if scalebar:
                try:
                    phys_size = f[path].attrs['size'][0]
                    px = f[path].attrs['shape'][0]
                    scalebar = ScaleBar(phys_size/px, f[path].attrs['unit'][0], location='lower right', font_properties={'size':25})
                    fig.add_artist(scalebar)
                except:
                    print("Error in the creation of the scalebar, please check that the attributes size and shape are correctly define for each datas channels.")

            fig.savefig(saving_path+path.rsplit('/')[-2]+'_'+str(image_name)+'.png')
            if verbose:
                print(filename.split('.')[0]+'_'+str(image_name)+'.png saved.')
            plt.close()
    if erase:
        import os
        os.remove(filename)
        erase = False
    return

#FUNCTION distortion_params_
## determine cumulative translation matrices for distortion correction.
#INPUTS:
## filename: name of hdf5 file containing data
## data_folder: folder searched for inputs. eg. 'datasets', or 'process/negative'
## selection: determines the name of folders or files to be used. Can be None (selects all), a string, or a list of strings. Default allows for correction based on topography in data from Asylum AFM.
## selection_depth: determines what level at which to look at a selection. Default allows for correction based on topography in data from Asylum AFM.
#OUTPUTS:
## null

def distortion_params_(filename, data_folder='datasets', selection = 'HeightRetrace', selection_depth = 2):
    path_lists = pt.initialise_process(filename, 'distortion_params', data_folder=data_folder, selection=selection, selection_depth=selection_depth)
    fineCheck = False
    tform21 = np.eye(2,3,dtype=np.float32)
    cumulative_tform21 = np.eye(2,3,dtype=np.float32)
    with h5py.File(filename, "a") as f:
        for i in range(len(path_lists)):
            if i == 0:
                pass
            else:
                img1 = img2cv((f[path_lists[i-1][0]]))
                img2 = img2cv((f[path_lists[i][0]]))
                tform21 = generate_transform_xy(img1, img2, tform21, fineCheck)
                cumulative_tform21[0,2]=cumulative_tform21[0,2]+tform21[0,2]
                cumulative_tform21[1,2]=cumulative_tform21[1,2]+tform21[1,2]
                print('Scan '+str(i)+' Complete. Cumulative Transform Matrix:')
                print(cumulative_tform21)
            pt.generic_write(f, cumulative_tform21, path_lists[i])
        

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
## tfinit: ???
## fineCheck: Initially used to force an "initial guess"
#OUTPUTS:
## Transformation images used to convert img_orig into img

def generate_transform_xy(img, img_orig, tfinit=None, fineCheck = False):
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

    if fineCheck == False:
        try:
            diff = np.Inf
            for i in range(-5,4):
                for j in range(-5,4):
                    warp_matrix[0,2] = 2*i
                    warp_matrix[1,2] = 2*j
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
        except:
            diff = np.Inf
            for i in range(-11,10):
                for j in range(-11, 10):
                    warp_matrix[0,2] = 2*i
                    warp_matrix[1,2] = 2*j
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
                
    else:
        diff = np.Inf
        for i in range(-60,-55):
            for j in range(-15, 15):
                warp_matrix[0,2] = 2*i
                warp_matrix[1,2] = 2*j
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

def distortion_correction_(filename, data_folder='datasets', selection=None, selection_depth=0, dm_data_folder = 'process/1-distortion_params', dm_selection=None, dm_selection_depth=0, cropping = True):
    dm_path_lists = pt.initialise_process(filename, None, data_folder=dm_data_folder, selection=dm_selection, selection_depth=dm_selection_depth)
    distortion_matrices = []
    with h5py.File(filename, "a") as f:
        for path in dm_path_lists[:]:
            distortion_matrices.append(np.copy(f[path]))
        xoffsets = []
        yoffsets = []
        for matrix in distortion_matrices:
            xoffsets.append(np.array(matrix[0,2]))
            yoffsets.append(np.array(matrix[1,2]))
    offset_caps = [np.max(xoffsets), np.min(xoffsets), np.max(yoffsets), np.min(yoffsets)]

    path_lists = pt.initialise_process(filename, 'distortion_correction', data_folder=data_folder, selection=selection, selection_depth=selection_depth)
    if len(path_lists)%len(dm_path_lists):
        print('Error: Images to be corrected are not a multiple of the amount of distortion matrices')
        return

    number_of_images_for_each_matrix = len(path_lists)//len(dm_path_lists)
    with h5py.File(filename, "a") as f:
        j = -1
        for i in range(len(path_lists)):
            if i%number_of_images_for_each_matrix == 0:
                j = j+1
            orig_image = f[path_lists[i][0]]
            if cropping == True:
                final_image = array_cropped(orig_image, xoffsets[j], yoffsets[j], offset_caps)
            else:
                final_image = array_expanded(orig_image, xoffsets[j], yoffsets[j], offset_caps)
            pt.generic_write(f, final_image, path_lists[i], 'source (distortion params)', dm_path_lists[j], 'distortion_params', distortion_matrices[j])

            
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