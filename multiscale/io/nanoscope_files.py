import numpy as np
import h5py

def convert_length_to_SI_unit(value, unit):
    unit_dic = {'am': 1e-18,
                'fm': 1e-15,
                'pm': 1e-12,
                'nm': 1e-9,
                '~m': 1e-6,
                'mm': 1e-3,
                'cm': 1e-2,
                'dm': 1e-1,
                'm': 1.,
                'km': 1e3}

    if unit not in unit_dic.keys():
        scale = 1
        print('The unit type is not supported!')
    else:
        scale = unit_dic[unit]
    return value * scale


def extract_scan_info_from_header(header):
    scan_header = header.split(b'\*Ciao scan list')[1].split(b'\*')[0]
    scan_header = scan_header.strip().split(b'\r\n')
    scan_dic = {}
    for line in scan_header:
        line = line.strip(b'\\')
        # discard lines starting with an '@' for now
        if line.startswith(b'@'):
            continue
        try:
            key, value = line.split(b': ')
        except ValueError:
            continue
        if key in ['Samps/line', 'Lines']:
            value = int(value)
        elif key in ['Rotate Ang.', 'Scan rate', 'Tip velocity']:
            value = float(value)
        elif key in ['Scan Size', 'X Offset', 'Y Offset']:
            value, unit = value.split(b' ')
            value = convert_length_to_SI_unit(float(value), unit)

        scan_dic[key] = value
    return scan_dic


def extract_image_infos_from_header(header):
    header_of_images = header.split(b'\*Ciao image list\r\n')[1:]
    image_infos = []
    for head in header_of_images:
        head = head.strip().split(b'\r\n')
        image_infos.append(extract_image_info_from_header(head))
    return image_infos


def extract_image_info_from_header(header):
    header_dic = {}
    for line in header:
        line = line.strip(b'\\')
        # discard lines starting with an '@' for now
        if line.startswith(b'@'):
            continue
        try:
            key, value = line.split(b': ')
        except ValueError:
            continue

        if key in ['Data offset', 'Data length', 'Bytes/pixel', 'Samps/line',
                   'Number of lines']:
            value = int(value)
        elif key in ['Scan Size']:
            value_x, value_y, unit = value.split(' ')
            value_x = convert_length_to_SI_unit(float(value_x), unit)
            value_y = convert_length_to_SI_unit(float(value_y), unit)
            value = np.array([value_x, value_y], dtype='float')

        header_dic[key] = value

    return header_dic


def load_nanoscope(filename):
    """
    Load a Nanoscope afm measurement files (.000, .001, etc.) and
    retrieve scan informations.

    Parameters
    ----------
    filename : string
        Filename

    Returns
    -------
    data : array-like
        image data
    scan_info : dictonary
        scan information
    image_infos : list
        image information
    header : dictonary
        information contained in the header
    """

    fn = open(filename, 'rb')

    file_str = fn.read()
    fn.close()
    # SEPARATE HEADER FROM DATA
    header = file_str.split(b'File list end\r\n')[0]
    # RETRIEVE SCAN PARAMETERS FROM FILE HEADER
    scan_info = extract_scan_info_from_header(header)
    image_infos = extract_image_infos_from_header(header)

    # extract the image data
    data_offset = int(image_infos[0][b'Data offset'].decode('iso-8859-1'))

    data_orig = np.frombuffer(file_str, dtype='<h', offset=data_offset)
    pixels = int(scan_info[b'Samps/line'].decode('iso-8859-1'))
    lines = int(scan_info[b'Lines'].decode('iso-8859-1'))
    data = np.zeros((len(image_infos), pixels, lines))
    start = 0
    for chan in range(len(image_infos)):
        tempX = int(image_infos[chan][b'Valid data len X'].decode('iso-8859-1'))
        tempY = int(image_infos[chan][b'Valid data len Y'].decode('iso-8859-1'))
        tempD = data_orig[start:start + tempX * tempY].reshape((tempY, tempX))
        data[chan][0:tempY, 0:tempX] = tempD
        start = start + tempX * tempY

    scan_info = {y.decode('iso-8859-1'): scan_info.get(y).decode('iso-8859-1') for y in scan_info.keys()}
    for i, k in enumerate(image_infos):
        image_infos[i] = {y.decode('iso-8859-1'): image_infos[i].get(y).decode('iso-8859-1') for y in
                          image_infos[i].keys()}
    header = header.decode('iso-8859-1')

    return data, scan_info, image_infos, header


def nanoscope2hdf5(filename):
    data, scan_info, image_infos, header = load_nanoscope(filename)
    with h5py.File(filename.replace('.', '_') + ".hdf5", "w") as f:
        typegrp = f.create_group("type")
        typegrp.create_dataset(filename.replace('.', '_'), data='Nanoscope')

        metadatagrp = f.create_group("metadata")
        metadatagrp.create_dataset(filename.replace('.', '_'), data=header)

        f.create_group("process")

        datagrp = f.create_group("datasets/" + filename.replace('.', '_'))

        for key in scan_info.keys():
            datagrp.attrs[key] = scan_info[key]
        # Get the name and trace orientation of each the channels
        nameChan = []
        for i in range(1, len(header.split('\@2:Image Data: '))):
            chan = header.split('\@2:Image Data: ')[i].split('\r\n')[0].split()[-1][1:-1]
            trace = header.split('\Line Direction: ')[i].split('\r\n')[0].split()[-1]
            full_name = chan + '_' + trace
            nameChan.append(full_name)

        for indx, name in enumerate(nameChan):
            dtst = datagrp.create_dataset(name, data=data[indx])
            for key in image_infos[indx].keys():
                dtst.attrs[key] = image_infos[indx][key]
