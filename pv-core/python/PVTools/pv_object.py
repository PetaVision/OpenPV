class PV_Object(list):
    def __init__(self, (data, header)):
        list.__init__(self, data)
        self.header = header

    def getActive(self):
        actList = []
        for frame in self:
            actList.append(np.count_nonzero(frame.values))
        return actList
