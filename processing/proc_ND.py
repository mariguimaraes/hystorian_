from processing import proc_tools as pt
import numpy as np
import h5py

def extract_hyst(filename):
    with h5py.File(filename, 'a') as f:
        
        notedict = {}
        for file in f['metadata'].keys(): 
            for k in f['metadata'][file]:
                tmp = k.decode('utf-8').split(":",1)
                try:
                    notedict[tmp[0]] = tmp[1]
                except:
                    pass

        for file in f['datasets'].keys():
            bias = f['datasets'][file]['Bias']
            waveform_amp = notedict["ARDoIVAmp"]
            waveform_freq = float(notedict["ARDoIVFrequency"])
            waveform_phase = float(notedict["ARDoIVArg2"])
            waveform_pulsetime = float(notedict["ARDoIVArg3"])
            if 'ARDoIVArg4' in notedict.keys():
                waveform_dutycycle = float(notedict["ARDoIVArg4"])
            else:
                waveform_dutycycle = 0.5

            waveform_delta = 1/float(notedict["NumPtsPerSec"])
            waveform_numbiaspoints = int(np.floor(waveform_delta*len(bias[1,1,:])/waveform_pulsetime))
            waveform_pulsepoints = int(waveform_pulsetime/waveform_delta)
            waveform_offpoints = int(waveform_pulsepoints*(1.0-waveform_dutycycle))
            
            for k in f['datasets'][file]:
                x,y,z = np.shape(f['datasets'][file][k])
                b = waveform_numbiaspoints
                dgroup = f.require_group('process/' + file + '/' + 'SSPFM_parameters/')
                dset_on = dgroup.require_dataset(k + '_on', (x,y,b), dtype=float)
                dset_off = dgroup.require_dataset(k + '_off', (x,y,b), dtype=float)


                for b in range(waveform_numbiaspoints):
                    start = b*waveform_pulsepoints+waveform_offpoints
                    stop = (b+1)*waveform_pulsepoints

                    var2 = stop-start+1
                    realstart = int(start+var2*.25)
                    realstop = int(stop-var2*.25)
                    f['process/' + file + '/' + 'SSPFM_parameters/'+k+'_on'][:,:,b] = \
                        np.nanmean(f['datasets'][file][k][:,:,realstart:realstop])

                    start = stop
                    stop = stop + waveform_pulsepoints*waveform_dutycycle

                    var2 = stop-start+1
                    realstart = int(start+var2*.25)
                    realstop = int(stop-var2*.25)
                    f['process/' + file + '/' + 'SSPFM_parameters/'+k+'_off'][:,:,b] = \
                        np.nanmean(f['datasets'][file][k][:,:,realstart:realstop])


def calc_hyst_params(bias, phase):
    biasdiff = np.diff(bias)
    up = np.sort(np.unique(np.hstack((np.where(biasdiff>0)[0],np.where(biasdiff>0)[0]+1))))
    dn = np.sort(np.unique(np.hstack((np.where(biasdiff<0)[0],np.where(biasdiff<0)[0]+1))))


    #UP leg calculations
    x = np.array(bias[up])
    y = np.array((phase[up]+360)%360)
    step_left_up = np.median(y[np.where(x==np.min(x))[0]])
    step_right_up = np.median(y[np.where(x==np.max(x))[0]])



    avg_x = []
    avg_y = []
    for v in np.unique(x):
        avg_x.append(v)
        avg_y.append(np.mean(y[np.where(x==v)[0]]))

    my_x = np.array(avg_x)[1:]+0*(avg_x[0]+avg_x[1])/2.0
    my_y = np.abs(np.diff(avg_y))

    coercive_volt_up = my_x[np.nanargmax(my_y)]

    #DOWN leg calculations
    x = np.array(bias[dn])
    y = np.array((phase[dn]+360)%360)
    step_left_dn = np.median(y[np.where(x==np.min(x))[0]])
    step_right_dn = np.median(y[np.where(x==np.max(x))[0]])

    avg_x = []
    avg_y = []
    for v in np.unique(x):
        avg_x.append(v)
        avg_y.append(np.mean(y[np.where(x==v)[0]]))

    my_x = np.array(avg_x)[1:]+0*(avg_x[0]+avg_x[1])/2.0
    my_y = np.abs(np.diff(avg_y))

    coercive_volt_dn = my_x[np.nanargmax(my_y)]

    return [coercive_volt_up, coercive_volt_dn, 0.5*step_left_dn+0.5*step_left_up, 0.5*step_right_dn+0.5*step_right_up]

def PFM_params_map(filename, phase = 1):
    
    with h5py.File(filename, 'a') as f:
        for file in f['process/'].keys():
            bias = f['process/' + file + '/' + 'SSPFM_parameters/Bias_on']
            if phase == 1:
                phase = f['process/' + file + '/' + 'SSPFM_parameters/Phase_on']
            else:
                phase = f['process/' + file + '/' + 'SSPFM_parameters/Phas2_on']
            x,y,z = np.shape(phase)
            list_values = ['coerc_pos', 'coerc_neg', 'step_left', 'step_right', 'imprint', 'phase_shift']
            f['process/' + file + '/' ].require_group('PFM_physical_parameters')
            for val in list_values:
                f['process/' + file + '/' + 'PFM_physical_parameters/'].require_dataset(val, (x,y), dtype=float)
            
            for xi in range(x):
                for yi in range(y):    
                    hyst_matrix = calc_hyst_params(bias[xi,yi,:],phase[xi,yi,:])
                    f['process/' + file + '/' + 'PFM_physical_parameters/coerc_pos'][xi,yi] = hyst_matrix[0] 
                    f['process/' + file + '/' + 'PFM_physical_parameters/coerc_neg'][xi,yi] = hyst_matrix[1]
                    f['process/' + file + '/' + 'PFM_physical_parameters/step_left'][xi,yi] = hyst_matrix[2]
                    f['process/' + file + '/' + 'PFM_physical_parameters/step_right'][xi,yi] = hyst_matrix[3]
                    f['process/' + file + '/' + 'PFM_physical_parameters/imprint'][xi,yi] = (hyst_matrix[0]+hyst_matrix[1])/2.0
                    f['process/' + file + '/' + 'PFM_physical_parameters/phase_shift'][xi,yi] = (hyst_matrix[3]-hyst_matrix[2])
