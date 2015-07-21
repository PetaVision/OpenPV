#########################################
##  plotError.py
##  Written by Dylan Paiton, William Shainin
##  Dec 31, 2014
##
#########################################

#TODO: Can we plan out the imports better so they are only imported once & when needed?
#      What is proper importing protocol?
import numpy as np
import matplotlib.pyplot as plt # only need if showPlot==True OR savePlot==true
import os                       # only needed if savePlot==true
import math

def plotSnrDbErr(preLayer,postLayer,showPlot=False,savePlot=False,saveName=''):
    # SNRdb = 10*log_10 ( image_pixel_variance / mean_squared_error )
    SNRdb = np.zeros((2,len(preLayer["values"])))
    for frame in range(len(preLayer["values"])):
        preImage  = np.squeeze(preLayer["values"][frame])
        postImage = np.squeeze(postLayer["values"][frame])
        preVar    = np.sum(np.square(preImage-np.mean(preImage)))/(len(np.ravel(preImage))-1)
        preMean   = np.sum(preImage)/len(np.ravel(preImage))
        mse       = np.sqrt(np.square(preImage-postImage).mean(axis=None))

        SNRdb[0,frame] = preLayer["time"][frame]
        SNRdb[1,frame] = 10*math.log10(preVar/mse)

    if showPlot:
        fig = plt.figure()
        plt.plot(SNRdb[0,:],SNRdb[1,:]) # plot(x,y)
        plt.xlabel('Time (dt)')
        plt.ylabel('SNRdb')
        plt.show(block=False) # If the code is run in interactive mode, show will block by default
    if savePlot:
        #TODO: Should be able to pass figure title?
        if len(saveName) == 0:
            fileName = 'plotSnrDbErr'
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

            fig = plt.figure()
            plt.plot(SNRdb[0,:],SNRdb[1,:])
            plt.xlabel('Time (dt)')
            plt.ylabel('SNRdb')
            plt.savefig(filePath+fileName+'.'+fileExt,bbox_inches='tight')

    return SNRdb

def plotPercErr(preLayer,postLayer,showPlot=False,savePlot=False,saveName=''):

    percErr = np.zeros((2,len(preLayer["values"])))
    for frame in range(len(preLayer["values"])):
        preImage  = np.squeeze(preLayer["values"][frame])
        postImage = np.squeeze(postLayer["values"][frame])

        percErr[0,frame] = preLayer["time"][frame]
        percErr[1,frame] = 100 * np.sqrt(np.sum(np.square(postImage))) / np.sqrt(np.sum(np.square(preImage)))

    if showPlot:
        fig = plt.figure()
        plt.plot(percErr[0,:],percErr[1,:]) # plot(x,y)
        plt.xlabel('Time (dt)')
        plt.ylabel('Percent Error')
        plt.show(block=False)
    if savePlot:
        #TODO: Should be able to pass figure title?
        if len(saveName) == 0:
            fileName = 'plotPercErr'
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

            fig = plt.figure()
            plt.plot(percErr[0,:],percErr[1,:])
            plt.xlabel('Time (dt)')
            plt.ylabel('Percent Error')
            plt.savefig(filePath+fileName+'.'+fileExt,bbox_inches='tight')

    return percErr 

