#########################################
##  plotActivity.py
##  Written by Dylan Paiton, William Shainin
##  Dec 31, 2014
##
#########################################

#TODO: Can we plan out the imports better so they are only imported once & when needed?
#      What is proper importing protocol?
import numpy as np
import matplotlib.pyplot as plt # only need if showPlot==True OR savePlot==true
import os                       # only needed if savePlot==true

def plotPercentActive(layer,showPlot=False,savePlot=False,saveName=''):
    numFrames     = len(layer["values"])
    percentActive = np.zeros(numFrames)
    numElements = layer["values"][0].shape[1] #TODO: only works if coo sparse matrix
    for frame in range(numFrames):
        numNotZero  = layer["values"][frame].nnz
        percentActive[frame] = numNotZero/numElements

    if showPlot:
        plt.figure()
        plt.plot(layer["time"],percentActive)
        plt.xlabel('Time (dt)')
        plt.ylabel('Percent Active')
        plt.ylim((0,1))
        plt.show(block=False)
    if savePlot:
        #TODO: Should be able to pass figure title & axis labels?
        if len(saveName) == 0:
            fileName = 'plotPercActive'
            fileExt  = 'png'
            filePath = './'
            saveName = filePath+fileName+'.'+fileExt
        else:
            seps     = saveName.split(os.sep)
            fileName = seps[-1]
            filePath = saveName[0:-len(fileName)]
            seps     = fileName.split(os.extsep)
            fileExt  = seps[-1]
            fileName = seps[0]
            if not os.path.exists(filePath):
                os.makedirs(filePath)

            plt.figure()
            plt.plot(layer["time"],percentActive)
            plt.xlabel('Time (dt)')
            plt.ylabel('Percent Active')
            plt.ylim((0,1))
            plt.show(block=False)
            plt.savefig(filePath+fileName+'.'+fileExt,bbox_inches='tight')


    return np.array((layer["time"],percentActive))


#TODO:
#def plotPercentChange(layer,showPlot=False,savePlot=False,saveName=''):
