from .readpvpfile import readpvpfile
#from .pv_object import PV_Object
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#FILE NOT TESTED

def interpret(arg):
   if type(arg) is str:
      arg = readpvpfile(arg)
   #assert type(arg) is PV_Object
   return arg

def getActive(data):
    numFrames = data["values"].shape(axis=0)
    actList = np.zeros((numFrames))
    #Can only be layer activity
    assert (data.header['filetype'] == 4 or data.header['filetype'] == 6)
    for frame in range(numFrames):
        allValues = data["values"]
        #dense
        if(data.header['filetype'] == 4):
            actList[frame] = np.count_nonzero(allValues[frame, :, :, :])
        else:
            actList[frame] = allValues.getrow(frame).nnz
    return actList

def getPercentActive(data):
    total = data.header['nx'] * data.header['ny'] * data.header['nf']
    return float(data.getActive()) / float(total)

def getError(data, *args):
    assert data.header['filetype'] == 4
    numDataFrames = data["values"].shape(axis=0)
    if args:
        outList = []
        for a in args:
            assert type(a) == type(data)
            assert a.header['filetype'] == 4
            numAFrames = a["values"].shape(axis=0)
            errList = np.zeros(min(numDataFrames, numAFrames))
            for frame in range(len(errList)):
                errList[frame] = np.linalg.norm((data["values"][frame, :, :, :] - a["values"][frame, :, :, :]))
            outList.append(errList)
        if len(outList) == 1:
            return outList[0]
        return outList
    else:
        errList = np.zeros(numDataFrames)
        for frame in range(numDataFrames):
            errList[frame] = np.linalg.norm(data["values"][frame, :, :, :])
        return errList

def view(data,frame=0):
   assert type(frame) is int
   data = interpret(data)

   if data.header['filetype'] == 4:
      plt.imshow(data["values"][frame, :, :, :],
                 interpolation='nearest')
      plt.axis('off')
      plt.show()

   if data.header['filetype'] == 5:
         axes = np.ceil(np.sqrt(data.header['numpatches']))
         for patch in range(data.header['numpatches']):
            plt.subplot(axes,axes,patch+1)
            plt.imshow(data["values"][frame, 0, patch, :, :, :],
                       interpolation='nearest')
            plt.axis('off')
         plt.show()

def showErrorPlot(image, *args):
   image = interpret(image)
   if args:
      for arg in args:
         plt.figure()
         plt.plot(getError(image, interpret(arg)))
         plt.show()
   else:
      plt.figure()
      plt.plot(getError(image))
      plt.show()

def showNumActivePlot(data):
   data = inpterpret(data)
   plt.figure()
   plt.plot(getActive(data))
   plt.show()

def showSparsityPlot(data):
   data = interpret(data)
   plt.figure()
   plt.plot(getPercentActive(data))
   plt.show()
