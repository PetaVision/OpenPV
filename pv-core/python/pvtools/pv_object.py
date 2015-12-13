import numpy as np
class PV_Object(list):
    def __init__(self, data, header):
        self.header = header
        list.__init__(self, data)

    def getActive(self):
        actList = np.zeros(len(self))
        if self.header['filetype'] == 5:
            return None
        for frame in range(len(self)):
            actList[frame] = np.count_nonzero(self[frame].values)
        if self.header['filetype'] == 6:
            actList = actList / 2
        return actList

    def getPercentActive(self):
        if self.header['filetype'] == 5:
            return None
        total = self.header['nx'] * self.header['ny'] * self.header['nf']
        actList = self.getActive() / float(total) 
        return actList
