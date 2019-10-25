from igor import binarywave

#==========================================
#IBW conversion

def correct_label(label):
    label = [x for x in label if x]  # Remove the empty lists
    label = label[0]  # Remove the unnecessary inception

    corrected_label = []

    for i in label:
        i = i.decode('UTF-8')
        if len(i) == 0:  # Skip empty channel names
            pass
        else:  # Correct the duplicate letters
            if 'cece' in i:
                corrected_label.append(i.replace('cece', 'ce'))
            elif 'aceace' in i:
                corrected_label.append(i.replace('aceace', 'ace'))
            else:
                corrected_label.append(i)
    corrected_label = [x.encode() for x in corrected_label]
    return corrected_label    
    
def ibw2hdf5(filename):
    try:
        tmpdata = binarywave.load(filename)['wave']
        note = tmpdata['note']

        label_list = filename.split('.')[-1]

        with h5py.File(filename.split('.')[0] + ".hdf5", "w") as f:
            file_type = get_type(filename)
            f.create_dataset("type", data=file_type)
            f.create_dataset("metadata", data=tmpdata['note'])
            f.create_dataset("channels/name", data=label_list)

            sizes = []
            for i, k in enumerate(label_list):
                sizes.append(tmpdata['wData'][:,:,i].shape)

            f.create_dataset("channelsdata/pxs", data=sizes)
            f.create_dataset("data", data=tmpdata['wData'])
        print('file successfully converted')
    except:
        print('error in the ibw->hdf5 conversion')
        
