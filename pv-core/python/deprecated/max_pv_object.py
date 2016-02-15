import numpy as np
class PV_Object(list):
    def __init__(self, data, header = None, name = None):
        self.header = header
        self.name = name
        list.__init__(self, data)

    def getActive(self):
        actList = np.zeros(len(self))
        assert (self.header['filetype'] == 5 or self.header['filetype'] == 6)
        for frame in range(len(self)):
            actList[frame] = np.count_nonzero(self[frame].values)
        if self.header['filetype'] == 6:
            actList = actList / 2
        return actList

    def getPercentActive(self):
        total = self.header['nx'] * self.header['ny'] * self.header['nf']
        return self.getActive() / float(total)

    def getError(self, *args):
        assert self.header['filetype'] == 4
        if args:
            outList = []
            for a in args:
                assert type(a) == type(self)
                assert a.header['filetype'] == 4
                errList = np.zeros(min(len(self),len(a)))
                for frame in range(len(errList)):
                    errList[frame] = np.linalg.norm((self[frame].values - a[frame].values))
                outList.append(errList)
            if len(outList) == 1:
                return outList[0]
            return outList
        
        else:
            errList = np.zeros(len(self))
            for frame in range(len(self)):
                errList[frame] = np.linalg.norm(self[frame].values)
            return errList
            
