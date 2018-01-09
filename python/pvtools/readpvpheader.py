import numpy as np
headerPattern = [('headersize', np.int32),
                 ('numparams', np.int32),
                 ('filetype', np.int32),
                 ('nx', np.int32),
                 ('ny', np.int32),
                 ('nf', np.int32),
                 ('numrecords', np.int32),
                 ('recordsize', np.int32),
                 ('datasize', np.int32),
                 ('datatype', np.int32),
                 ('nxprocs', np.int32),
                 ('nyprocs', np.int32),
                 ('nxExtended', np.int32),
                 ('nyExtended', np.int32),
                 ('kx0', np.int32),
                 ('ky0', np.int32),
                 ('nbatch', np.int32),
                 ('nbands', np.int32),
                 ('time', np.float64)]

extendedHeaderPattern = headerPattern + [('nxp', np.int32),
                                         ('nyp', np.int32),
                                         ('nfp', np.int32),
                                         ('wMin', np.float32),
                                         ('wMax', np.float32),
                                         ('numpatches', np.int32)]

def readpvpheader(fileStream):
    stringFlag = False
    if type(fileStream) == str:
        stringFlag = True
        fileStream = open(fileStream,'rb')

    header = np.fromfile(fileStream,np.dtype(headerPattern),1)
    if header['numparams'] == 26:
        fileStream.seek(0)
        header = np.fromfile(fileStream,np.dtype(extendedHeaderPattern),1)

    if stringFlag:
        fileStream.close()

    return dict(zip(header.dtype.names,header[0]))
