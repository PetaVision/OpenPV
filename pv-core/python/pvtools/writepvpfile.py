# Converts a PV_Ovject into a .pvp file.


import numpy as np

def writepvpfile(pvobject, filename):

# First, write header:

    # To get ordered list of header params
    from .readpvpheader import headerPattern, extendedHeaderPattern
    if pvobject.header['numparams'] == 26:
        headerPattern = extendedHeaderPattern

    with open(filename, 'wb') as stream:
        # Header entries is a list of tuples,
        # [0] being the name (like 'filetype' or 'nx')
        # and [1] being the dtype.
        for headerEntry in headerPattern:
            stream.write(headerEntry[1](pvobject.header[headerEntry[0]]))

# Second, write data:
        if pvobject.header['filetype'] == 1:
            pass
        elif pvobject.header['filetype'] == 2:
            pass
        elif pvobject.header['filetype'] == 3:
            pass 
        elif pvobject.header['filetype'] == 4:
            for dataFrame in pvobject:
                stream.write(dataFrame.time)
                stream.write(dataFrame.values)
        elif pvobject.header['filetype'] == 5:
            pass
        elif pvpbject.header['filetype'] == 6:
            pass




