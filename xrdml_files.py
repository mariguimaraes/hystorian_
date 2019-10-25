import xrdtools

#==========================================
#XRDML conversion

def xrdml2hdf5(filename):
    with open(filename, 'r') as f:
        contents = f.read()

    counts = contents.split('<counts unit="counts">')[-1].split('</counts>')[0].split()
    cnts = list(map(float, counts))

    positions = contents.split('<dataPoints>')[-1].split('<beamAttenuationFactors>')[0]
    theta=contents.split('<positions axis="2Theta" unit="deg">')[-1].split('</positions>')[0]
    angles = []
    for i in theta.split():
        angles.append(float(re.findall(r"[-+]?\d*\.\d+|\d+",i)[0]))

    theta_list = np.arange(angles[0], angles[1], (angles[1] - angles[0])/len(cnts))

    with h5py.File(filename.split('.')[0] + ".hdf5", "w") as f:
        file_type = get_type(filename)
        f.create_dataset("type", data=file_type)
        f.create_dataset("metadata", data=contents)
        f.create_dataset("channels/name", data=['angle'.encode(), 'intensity'.encode()])
        f.create_dataset("data", data=(theta_list,cnts))
        
        def debug_print(dstr):
    if not dstr.startswith("Info"):
        print(dstr)
    