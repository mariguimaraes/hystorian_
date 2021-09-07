try:
    from matplotlib_scalebar.scalebar import ScaleBar
    scaleBarBool = True
except ImportError:
    print('Matplotlib_scalebar was not found, please install the package or you will not be able to'
          ' add scalebar to your images.')
    scaleBarBool = False

import numpy as np
import h5py
import matplotlib.pyplot as plt


def save_image(data,
               size=None,
               ticks=True,
               labelsize=16,
               full_range=False,
               std_range=3,
               cmap='inferno',
               colorbar=True,
               scalebar=False,
               physical_size=(0, 'unit'),
               source_scale_m_per_px=None,
               show=False,
               save=True,
               image_name=None,
               saving_path='',
               source_path=None,
               verbose=False,
               colorm=None):

    '''
    Wrapper around plt.imshow() and plt.savefig() to uniformize image saving

    Parameters
    ----------
    data: array-like
        array-like containing the image (1-channel)
    size: tuple (int,int), optional
        size in pxs of the image
    ticks: bool, optional
        shows ticks if set to True. (default is False)
    labelsize: int, optional
        setup the px size of the labels. (default: 16)
    full_range: bool, optional
        use the whole data scale if set to True, or use [mean-std_range*std, mean+std_range*std].
        (default: True)
    std_range: int or float, optional
        the number of standard deviation to use in case full_range is set to false. (default: 3)
    cmap: str, optional
        choose the colormap, see matplotlib for the list of available colormap (default: 'inferno')
    colorbar: bool, optional
        allows to remove the colorbar if set to false. (default: True)
    scalebar: bool, optional
        allows to show a scalebar, need a value in physical size. (Default: False)
    physical_size: tuple (float, str)
        physical size of the image to define the scalebar.
    source_scale_m_per_px:
        attempts to directly grab scale if attrs are provided (default: None)
    show: bool, optional
        If set to true, use plt.show() to show the image (default: False)
    save: bool, optional
        If set to true, use plt.imsave() to save the image (default: True)
    image_name: str, optional
        filename to use while saving the image. If set to None will try to use source_path to
        generate a filename, if no source_path is given, will use 'image' (default: None)
    saving_path: str, optional
        path where to save the file. (default: '')
    source_path: str, optional
        If set, and image_name not set, this variable will be used to generate the file name
        (default: None)
    verbose: bool, optional
        if True, print a line when the image is saved. (default: false)
    Returns
    -------
    None
    '''

    #Remove later
    if colorm:
        print('Use of \'colorm\' is depracated; use \'cmap\' instead')
        cmap = colorm

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
    pos = plt.imshow(data, vmin=v_min, vmax=v_max, cmap=cmap)

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
                print("Error in the creation of the scalebar, please check that the attribute's"
                      " size and shape are correctly define for each data channel.")
                raise
        else:
            print("Scalebar package is not installed, please install it if you want to add a"+
                  " scalebar to your image")
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

    '''
    NOTE - This function is not made to be used by m_apply, this is indicated by the trailing
    underscore. Used on data processed using twodim.calc_hyst_params() to plot and save an image 
    containing maps of the 6 hysteresis parameters : coercive voltage (up and down), step (left and
    right), imprint and phase shift.
    
    Parameters
    ----------
    filename: str
        hdf5 file containing the SSPFM parameters extracted using twodim.calc_hyst_params()
    PATH: str
        path inside the hdf5 file to the datas
    size: tuple (int,int), optional
        size in pxs of the image
    ticks: bool, optional
        shows ticks if set to True. (default is False)
    labelsize: int, optional
        setup the px size of the labels. (default: 16)
    colorbar: bool, optional
        allows to remove the colorbar if set to false. (default: True)
    scalebar: bool, optional
        allows to show a scalebar, need a value in physical size. (Default: False)
    show: bool, optional
        If set to true, use plt.show() to show the image (default: False)
    save: bool, optional
        If set to true, use plt.imsave() to save the image (default: True)
    image_name: str, optional
        filename to use while saving the image. If set to None will try to use source_path to
        generate a filename, if no source_path is given, will use 'image' (default: None)
    saving_path: str, optional
        path where to save the file. (default: '')
    source_path: str, optional
        If set, and image_name not set, this variable will be used to generate the file name
        (default: None)
    verbose: bool, optional
        if True, print a line when the image is saved. (default: false)
    Returns
    -------
        None
    '''

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

        
def plot_RSM(qx, qz, intensity, filename='RSM', xlim = [], ylim = []):
    """
    Saves an image from processed reciprocal space map data
      
    Parameters
    ----------
    qx : array
        in-plane reciprocal space vector
    qz : array
        out-of-plane reciprocal space vector
    intensity : array
        intensity at the site defined by qx, qz
    filename : string
        name of the output file
    xlim : list
        plot xlim
    ylim : list
        plot ylim
    
    Returns
    ------
        None
    """
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
    
    
def to_csv(data,
           file_name=None,
           saving_path='',
           source_path=None):
    """
    Exports a dataset into a .csv file
      
    Parameters
    ----------
    data : array
        data to be exported
    file_name : 2d array
        name of the output file. By default, names file 'data'
    saving_path : string
        path to save destination
    source_path : string
        for compatibility with source parameter in m_apply;
        used to name the output file
    
    Returns
    ------
        None
    """
    if file_name is None:
        if source_path is not None:
            file_name = source_path.replace('/', '_')
        else:
            file_name = 'data'
    if saving_path != '':
        if saving_path[-1] != '/':
            saving_path = saving_path + '/'
    np.savetxt(saving_path+str(file_name) + '.csv', data, delimiter=',')
    return