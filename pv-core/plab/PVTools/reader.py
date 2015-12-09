def readpvpheader(filename):
    import numpy as np
    header_names = ['headersize','numparams','filetype',
                    'nx','ny','nf','numrecords','recordsize',
                    'datasize','datatype','nxprocs','nyprocs',
                    'nxGlobal','nyGlobal','kx0','ky0','nbatch',
                    'nbands']
    fid = open(filename,'rb')
    head = zip(header_names, np.fromfile(fid,dtype=np.int32,count=18))
    head.append(('time', np.fromfile(fid,dtype=np.float64,count=1)[0]))
    if head[2][1] == 2:
        print("pvp file type 2 is not yet supported")
    if head[2][1] == 3:
        print("pvp file type 3 is not yet supported")
    if head[2][1] == 5:
        head.append(('nxp', np.fromfile(fid,dtype=np.int32,count=1)[0]))
        head.append(('nyp', np.fromfile(fid,dtype=np.int32,count=1)[0]))
        head.append(('nfp', np.fromfile(fid,dtype=np.int32,count=1)[0]))
        head.append(('wMin', np.fromfile(fid,dtype=np.float32,count=1)[0]))
        head.append(('wMax', np.fromfile(fid,dtype=np.float32,count=1)[0]))
        head.append(('numpatches', np.fromfile(fid,dtype=np.int32,count=1)[0]))
    if head[2][1] == 6:
        print("pvp file type 6 is not yet supported")
    return dict(head)


def readpvpfile(filename):
    from oct2py import octave
    import numpy as np
    import re
    octave.addpath(re.match('(.*)(plab)',__file__).group(0) + '/mlab/util')
    h = readpvpheader(filename)
    raw_data = octave.readpvpfile(filename)
    data = []
    if h['filetype'] == 2:
        print("pvp file type 2 is not yet supported")
        return
    if h['filetype'] == 3:
        print("pvp file type 3 is not yet supported")
        return
    if h['filetype'] == 4:
        shape = [h['nx'],h['ny'],h['nf']]
        for frame in raw_data:
            data.append( {'time': frame['time'][0][0][0], 'values': np.reshape(np.array(frame['values'][0]),shape) } )
    if h['filetype'] == 5:
        shape = [h['nxp'],h['nyp'],h['nfp'],h['numpatches']]
        for frame in raw_data:
            data.append( {'time': frame['time'], 'values': np.reshape(np.array(frame['values']),shape) } )
    if h['filetype'] == 6:
        print("pvp file type 6 is not yet supported")
        return

    return data
