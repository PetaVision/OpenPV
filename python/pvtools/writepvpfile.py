from pvpFile import pvpOpen

#Convenience function for new and improved pvpOpen
def writepvpfile(filename, data, shape=None, useExistingHeader=False):
    f = pvpOpen(filename, 'w')
    f.write(data, shape, useExistingHeader)
    f.close()
