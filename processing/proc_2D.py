import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar

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
            data = f['datas']

            for k in data.keys():
                plt.figure(figsize=size)
                plt.tick_params(labelsize=labelsize)
                if 'Phase' in k:
                    colorm = 'inferno'
                    if np.min(data[k]) < 0:
                        offsetdata = data[k] + np.min(data[k])
                    else:
                        offsetdata = data[k] - np.min(data[k])
                    v_min = -180
                    v_max = 180
                else:
                    colorm = 'afmhot'
                    if np.min(data[k]) < 0:
                        offsetdata = data[k] + np.min(data[k])
                    else:
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
 
                plt.savefig(saving_path+filename.split('.')[0]+'_'+str(k)+'.png')
                if verbose:
                    print(filename.split('.')[0]+'_'+str(k)+'.png saved.')
                plt.close()
    return