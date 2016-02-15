# Converts a PV_Object into a .pvp file.

import numpy as np

def writepvpfile(pvObject, filename):

    # To get ordered list of header params
    from .readpvpheader import headerPattern, extendedHeaderPattern
    if pvObject.header['numparams'] == 26:
        headerPattern = extendedHeaderPattern

    with open(filename, 'wb') as stream:
        if pvObject.header['filetype'] == 1:
            print('Filetype 1 not yet supported for write pvp')
        elif pvObject.header['filetype'] == 2:
            print('Filetype 2 not yet supported for write pvp')
        elif pvObject.header['filetype'] == 3:
            print('Filetype 3 not yet supported for write pvp')

        elif pvObject.header['filetype'] == 4:
            # Tested as of 1/29/16
            for headerEntry in headerPattern:
                stream.write(headerEntry[1](pvObject.header[headerEntry[0]]))
            for dataFrame in pvObject:
                stream.write(dataFrame.time)
                stream.write(dataFrame.values)

        elif pvobject.header['filetype'] == 5:
            # Untested as of 1/29/16
            # Type 5's have a header in each frame
            for dataFrame in pvObject:
                for headerEntry in headerPattern:
                    stream.write(headerEntry[1](pvObject.header[headerEntry[0]]))
                stream.write(dataFrame.time)
                stream.write(dataFrame.values)

        elif pvpbject.header['filetype'] == 6:
            # Untested as of 1/29/16
            # Copied from filetype 4
            for headerEntry in headerPattern:
                stream.write(headerEntry[1](pvObject.header[headerEntry[0]]))
            for dataFrame in pvObject:
                stream.write(dataFrame.time)
                stream.write(dataFrame.values)




