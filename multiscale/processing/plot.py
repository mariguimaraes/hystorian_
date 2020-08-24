try:
    from matplotlib_scalebar.scalebar import ScaleBar
    scaleBarBool = True
except ImportError:
    print('Matplotlib_scalebar was not found, please install the package.')
    scaleBarBool = False

import numpy as np
import h5py
import matplotlib.pyplot as plt
    


#   FUNCTION save_image
# Saves one .png image to the current directory, or a chosen folder
#   INPUTS:
# data: A 2-D array which will be converted into a png image.
# size (default: None): Dimension of the saved image. If none, the image is set to have one pixel
#     per data point at 100 dpi
# ticks (default: True): Generate tick labels
# labelsize (default: 16): Size of the text in pxs
# full_range (default: False): Show whether to show the image as is, or to remove offset and use
#     std to figure out range
# std_range (default: 3): Range (in std) around the mean that is plotted
# colorm (default: 'inferno'): colormap to be used for image
# colorbar (default: True): option to generate a colour bar
# scalebar (default: False): if True, add a scalebar to the image, requires three attributes :
#     shape, which define the pixel size of the image
#     size, which gives the phyiscal dimension of the image
#     unit, which give the physical unit of size
# physical_size (default: (0, 'unit')): physical size of the image used when generating the scalebar
# source_scale_m_per_px (default: None): attempts to directly grab scale if attrs are provided
# show (default: False): if True, the image is displayed in the kernel
# save (default: True): determines if the image should be saved
# image_name (default: None): name of the image that is saved. By default, tries to pull name from
#     source_path. If this cannot be done, sets name to 'image'
# saving_path (default: ''): The path to the folder where to save the image
# source_path (default: None): if set, and image_name not set, this variable will be used to
#     generate the file name
# verbose (default: False): if True, print a line each time a image is saved.
#   OUTPUTS:
# null

def save_image(data,
               size=None,
               ticks=True,
               labelsize=16,
               full_range=False,
               std_range=3,
               colorm='inferno',
               colorbar=True,
               scalebar=False,
               physical_size=(0, 'unit'),
               source_scale_m_per_px=None,
               show=False,
               save=True,
               image_name=None,
               saving_path='',
               source_path=None,
               verbose=False):
    # Generate size of image frame
    if size is None:
        figsize = (np.array(np.shape(data)) / 100)[::-1]
        if figsize[0] < 3:
            scale_factor = np.ceil(3 / figsize[0])
            figsize = scale_factor * figsize
        fig = plt.figure(frameon=False, figsize=figsize, dpi=100)
    else:
        fig = plt.figure(figsize=size)

    # Generate ticks
    if ticks:
        plt.tick_params(labelsize=labelsize)
    else:
        plt.xticks([])
        plt.yticks([])

    # Set min and max:
    if data.dtype == 'bool':
        data = data.astype(int)
    if full_range:
        v_min = np.nanmin(data)
        v_max = np.nanmax(data)
    else:
        data = data - np.nanmin(data)
        mean_val = np.nanmean(data)
        std_val = np.nanstd(data)
        v_min = mean_val - std_range * std_val
        v_max = mean_val + std_range * std_val

    # Plot image
    pos = plt.imshow(data, vmin=v_min, vmax=v_max, cmap=colorm)

    # Generate colourbar
    if colorbar:
        cbar = plt.colorbar(pos, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=labelsize)
    plt.tight_layout()

    # Generate scalebar
    if scalebar:
        if scaleBarBool:
            try:
                if source_scale_m_per_px is None:
                    phys_size = physical_size[0]
                    px = np.shape(data)[0]
                    scalebar = ScaleBar(phys_size / px, physical_size[1], location='lower right',
                                        font_properties={'size': labelsize})
                else:
                    scalebar = ScaleBar(source_scale_m_per_px, 'm', location='lower right',
                                        font_properties={'size': labelsize})
                plt.gca().add_artist(scalebar)
            except:
                print("Error in the creation of the scalebar, please check that the attribute's\
                            size and shape are correctly define for each data channel.")
                raise
        else:
            print("Scalebar package is not installed, please install it if you want to add a scalebar to your image")
    # Generate ouputs:
    if save:
        if image_name is None:
            if source_path is not None:
                image_name = source_path.replace('/', '_')
            else:
                image_name = 'image'
        if saving_path != '':
            if saving_path[-1] != '/':
                saving_path = saving_path + '/'
        fig.savefig(saving_path + str(image_name) + '.png')
    if show:
        plt.show()
    if verbose:
        print(str(image_name) + '.png saved.')

    plt.close()
    return


#   FUNCTION plot_hysteresis_parameters_
# Saves one .png image containing the map of the 6 hysteresis parameters : coercive voltage (up and down),
# step (left and right), imprint and phase shift.
#   INPUTS:
# filename: hdf5 file containing a SSPFM map
# PATH: path to the folder containing the map of the hysteresis parameters
# size (default: None): Dimension of the saved image. If none, the image is set to have one pixel
#     per data point at 100 dpi
# ticks (default: True): Generate tick labels
# labelsize (default: 16): Size of the text in pxs
# colorbar (default: True): option to generate a colour bar
# show (default: False): if True, the image is displayed in the kernel
# save (default: True): determines if the image should be saved
# image_name (default: None): name of the image that is saved. By default, tries to pull name from
#     source_path. If this cannot be done, sets name to 'image'
# saving_path (default: ''): The path to the folder where to save the image
# source_path (default: None): if set, and image_name not set, this variable will be used to
#     generate the file name
# verbose (default: False): if True, print a line each time a image is saved.
#   OUTPUTS:
# null

def plot_hysteresis_parameters_(filename, PATH,
                                size=None,
                                ticks=True,
                                labelsize=16,
                                colorbar=True,
                                show=False,
                                save=True,
                                image_name=None,
                                saving_path='',
                                source_path=None,
                                verbose=False):
    print(PATH)
    if size is None:
        fig = plt.figure(figsize=(20, 30))
    else:
        fig = plt.figure(figsize=(size[0], size[1]))
    # Generate ticks
    if ticks:
        plt.tick_params(labelsize=labelsize)
    else:
        plt.xticks([])
        plt.yticks([])

    with h5py.File(filename) as f:
        plt.subplot(3, 2, 1)
        plt.title('Negative Coercive field')
        print(PATH + '/coerc_neg')
        plt.imshow(f[PATH + '/coerc_neg'], cmap='Blues')
        if colorbar:
            plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(3, 2, 2)
        plt.title('Positive Coercive field')
        plt.imshow(f[PATH + '/coerc_pos'], cmap='Reds')
        if colorbar:
            plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(3, 2, 3)
        plt.title('Left step')
        plt.imshow(f[PATH + '/step_left'], cmap='Greys')
        if colorbar:
            plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(3, 2, 4)
        plt.title('Right step')
        plt.imshow(f[PATH + '/step_right'], cmap='Greys')
        if colorbar:
            plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(3, 2, 5)
        plt.title('Imprint')
        plt.imshow(f[PATH + '/imprint'], cmap='Greys')
        if colorbar:
            plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(3, 2, 6)
        plt.title('Phase shift')
        plt.imshow(f[PATH + '/phase_shift'], cmap='Greys')
        if colorbar:
            plt.colorbar(fraction=0.046, pad=0.04)

    if save:
        if image_name is None:
            if source_path is not None:
                image_name = source_path.replace('/', '_')
            else:
                image_name = 'image'
        if saving_path != '':
            if saving_path[-1] != '/':
                saving_path = saving_path + '/'
        fig.savefig(saving_path + str(image_name) + '.png')
    if show:
        plt.show()
    if verbose:
        print(str(image_name) + '.png saved.')

        
#   FUNCTION plot_RSM
# Saves an image from processed reciprocal space map data
#   INPUTS:
# qx: in-plane reciprocal space vector
# qz: out-of-plane reciprocal space vector
# intensity: intensity at the site defined by qx, qz
# filename (default: 'RSM'): name of the output file
# xlim (default: []): plot xlim
# ylim (default: []): plot ylim
#   OUTPUTS:
# null
        
def plot_RSM(qx, qz, intensity, filename='RSM', xlim = [], ylim = []):
    x = qx.flatten()
    y = qz.flatten()
    z = np.log(intensity.flatten())
    plt.figure(figsize=(10,10))
    plt.tricontourf(x, y, z, levels=100, cmap='jet')
    edges_qx = []
    edges_qx.extend(qx[1,1:-2])
    edges_qx.extend(qx[1:-2,-2])
    edges_qx.extend(list(reversed(qx[-2,1:-1]))[:-1])
    edges_qx.extend(list(reversed(qx[1:-1,1]))[:-1])
    edges_qz = []
    edges_qz.extend(qz[1,1:-2])
    edges_qz.extend(qz[1:-2,-2])
    edges_qz.extend(list(reversed(qz[-2,1:-1]))[:-1])
    edges_qz.extend(list(reversed(qz[1:-1,1]))[:-1])
    min_i = np.argmin(edges_qx)
    max_i = np.argmax(edges_qx)
    if max_i<min_i:
        max_i, min_i = min_i, max_i
    qx_bound_1 = edges_qx[min_i:max_i+1]
    qz_bound_1 = edges_qz[min_i:max_i+1]
    qx_bound_2 = edges_qx[max_i:]
    qx_bound_2.extend(edges_qx[:min_i+1])
    qz_bound_2 = edges_qz[max_i:]
    qz_bound_2.extend(edges_qz[:min_i+1])
    centre_arg = int(len(qx_bound_1)/2)
    
    if qz_bound_1[centre_arg]<qz_bound_2[centre_arg]:
        qz_bound_1, qz_bound_2 = qz_bound_2, qz_bound_1 
        qx_bound_1, qx_bound_2 = qx_bound_2, qx_bound_1 
        
    linewidth=3
    plt.plot(qx_bound_1, qz_bound_1, 'k', linewidth=linewidth)
    plt.plot(qx_bound_2, qz_bound_2, 'k', linewidth=linewidth)
    
    qx_low_bound = qx_bound_1.copy()
    qx_high_bound = qx_bound_2.copy()
    qz_low_bound = qz_bound_1.copy()
    qz_high_bound = qz_bound_2.copy()
    
    if qx_low_bound [0]<qx_low_bound [-1]:
        qx_low_bound.insert(0,np.min(qx))
        qx_low_bound.append(np.max(qx))
    else:
        qx_low_bound.insert(0,np.max(qx))
        qx_low_bound.append(np.min(qx))    
        
    if qx_high_bound[0]<qx_high_bound[-1]:
        qx_high_bound.insert(0,np.min(qx))
        qx_high_bound.append(np.max(qx))
    else:
        qx_high_bound.insert(0,np.max(qx))
        qx_high_bound.append(np.min(qx))
        
    qz_low_bound.insert(0, qz_low_bound[0])
    qz_low_bound.append(qz_low_bound[-1])
    qz_high_bound.insert(0, qz_high_bound[0])
    qz_high_bound.append(qz_high_bound[-1])
    
    top_bound = np.zeros_like(qz_high_bound)+np.max(qz)
    bot_bound = np.zeros_like(qz_low_bound)+np.min(qz)
    plt.tick_params(labelsize=18)
    plt.fill_between(qx_low_bound, qz_low_bound, top_bound, facecolor='w')
    plt.fill_between(qx_high_bound, qz_high_bound, bot_bound, facecolor='w')
    plt.xlabel(r'$Q_x (1/{\AA})$', fontsize=24)
    plt.ylabel(r'$Q_z (1/{\AA})$', fontsize=24)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_ylabel('log10(Intensity)', fontsize=18)
    if not xlim:
        plt.xlim(np.min(qx_bound_1), np.max(qx_bound_1))
    else:
        plt.xlim(xlim[0], xlim[1])
    if not ylim:
        plt.ylim(np.min(qz_bound_1), np.max(qz_bound_1))
    else:
        plt.xlim(ylim[0], ylim[1])
    plt.savefig(filename+'.png')
    plt.show()
    plt.close()