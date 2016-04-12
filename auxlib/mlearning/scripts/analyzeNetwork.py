import pvtools as pv
import numpy as np
import pdb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

softmax = True
outputDir = "/home/sheng/mountData/KITTI/KITTI_Deep_2X3frames/softmax_benchmark0/"
plotsDir = outputDir + "/outplots/"
estDir = plotsDir + "/visOut/"
visEstLayer = "GroundTruthWTA.pvp"
visGtLayer = "GroundTruthDownsample.pvp"

if softmax:
    estLayer = "est.pvp"
else:
    estLayer = "GroundTruthReconS1.pvp"

gtLayer = "GroundTruthBin.pvp"
startFrame = 600


if not os.path.exists(plotsDir):
    os.makedirs(plotsDir)
if not os.path.exists(estDir):
    os.makedirs(estDir)

visEst = pv.readpvpfile(outputDir + visEstLayer)
visGt = pv.readpvpfile(outputDir + visGtLayer)

est = pv.readpvpfile(outputDir + estLayer)
gt = pv.readpvpfile(outputDir + gtLayer)

estVals = est["values"]
(numFrames, gtNy, gtNx, gtNf) = est["values"].shape
gtVals = np.array(gt["values"].todense()).reshape((numFrames, gtNy, gtNx, gtNf))

#Find valid indices
validIdx = np.nonzero(visGt["values"])

outPlotVals = np.zeros(numFrames)

for f in range(numFrames):
    for y in range(gtNy):
        for x in range(gtNx):
            #For DNC areas
            if visGt["values"][f, y, x, 0] != 0:
                if softmax:
                    outPlotVals[f] -= np.sum(gtVals[f, y, x, :] * np.log(estVals[f, y, x, :]))
                else:
                    accumVal = np.sum(np.power(gtVals[f, y, x, :] - estVals[f, y, x, :], 2))
                    outPlotVals[f] += accumVal

#Plot energy vs time
plt.plot(outPlotVals)
filename = plotsDir + "/costVsTime.png"
plt.savefig(filename)

f, axarr = plt.subplots(2, sharex=True)

for fi in range(startFrame, numFrames):
    print "Frame " + str(fi) + " out of " + str(numFrames)
    visEstVals = visEst["values"][fi, :, :, 0]
    visGtVals = visGt["values"][fi, :, :, 0]
    #Normalize based on maximum value of gt
    maxVal = np.max(visGtVals)
    axarr[0].imshow(visGtVals, vmin=0, vmax=maxVal)
    axarr[0].set_title('GT')
    axarr[1].imshow(visEstVals, vmin=0, vmax=maxVal)
    axarr[1].set_title('Est')
    filename = estDir + "/frame" + str(fi)
    plt.savefig(filename)
    plt.cla()

