from pvpFile import pvpOpen

#Convenience function for new and improved pvpOpen
def readpvpfile(filename,
                progressPeriod=0,
                lastFrame=-1,
                startFrame=0,
                skipFrames=1):

    f = pvpOpen(filename, 'r')
    out = f.read(startFrame, lastFrame, skipFrames, progressPeriod)
    f.close()
    return out
