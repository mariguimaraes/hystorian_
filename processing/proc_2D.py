import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
from . import proc_tools as pt
import cv2

#FUNCTION save_image 
#INPUTS:
## filename : The hdf5 file containing the data you want to convert into png
## scalebar (default: False): Add a scalebar to the image, requires three attributes : 
##                                                 shape, which define the pixel size of the image
##                                                 size, which gives the phyiscal dimension of the image
##                                                 unit, which give the physical unit of size
## size (default: (10,10)) : Dimension of the saved image
## labelsize (default: 25) : Size of the text in pxs
## std_range (default: 3) : Range around the mean for the colorscale, alternatively the value can be "full", to take the full range.
## saving_path (default: '') : The path to the folder where to save the image
## verbose (default: False) : if True, print a line each time a image is saved.
## Output : N png images, where N is the number of datas channels in the hdf5 file.

def save_image(filename, scalebar=False, size=(10,10), labelsize=25, std_range=3, saving_path='', verbose=False): 
    if filename.split('.')[-1] != 'hdf5':
        print("File extension is not hdf5, please convert it before using this function")
    else:
        with h5py.File(filename, "r") as f:
            data_folder = f['datasets']
            
            for k_folder in data_folder.keys():
            
                data = data_folder[k_folder]

                for k in data.keys():
                    plt.figure(figsize=size)
                    plt.tick_params(labelsize=labelsize)
                    if 'Phase' in k:
                        colorm = 'inferno'
                        offsetdata = data[k] - np.min(data[k])
                        v_min = -180
                        v_max = 180
                    else:
                        colorm = 'afmhot'
                        offsetdata = data[k] - np.min(data[k])
                        mean_val = np.mean(data[k])
                        std_val = np.std(data[k])
                        v_min = mean_val - std_range*std_val
                        v_max = mean_val + std_range*std_val
                    if std_range == "full":
                        pos = plt.imshow(offsetdata, cmap=colorm)
                    else:
                        try:
                            pos = plt.imshow(offsetdata, vmin=v_min, vmax=v_max, cmap=colorm)
                        except:
                            print("error in the min, max for the image, whole range is used.")
                            pos = plt.imshow(offsetdata, cmap=colorm)
                    cbar = plt.colorbar(pos,fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=25) 
                    plt.tight_layout()
                    #plt.ylim(plt.ylim()[::-1])
                    plt.axis('off')
                    if scalebar:
                        try:
                            phys_size = data[k].attrs['size'][0]
                            px = data[k].attrs['shape'][0]
                            scalebar = ScaleBar(phys_size/px, data[k].attrs['unit'][0], location='lower right', font_properties={'size':25})
                            plt.gca().add_artist(scalebar)
                        except:
                            print("Error in the creation of the scalebar, please check that the attributes size and shape are correctly define for each datas channels.")

                    plt.savefig(saving_path+str(data).split('/')[-1].split('\"')[0]+'_'+str(k)+'.png')
                    if verbose:
                        print(filename.split('.')[0]+'_'+str(k)+'.png saved.')
                    plt.close()
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