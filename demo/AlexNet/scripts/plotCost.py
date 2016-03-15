import pvtools as pv
import numpy as np
import pdb
import matplotlib.pyplot as plt

#Parameters
simDir = "/home/sheng/workspace/OpenPV/demo/AlexNet/trainOutput/"
outDir = "outPlots/"
estPvpFilename = simDir + "a18_est.pvp"
gtPvpFilename = simDir + "a5_gt.pvp"
batchSize = 128

#Get pvp files
estData = pv.readpvpfile(estPvpFilename, skipFrames = 10)
gtData = pv.readpvpfile(gtPvpFilename, skipFrames = 10)

#Reduce dimensions
estValues = np.squeeze(estData["values"])
gtValue = np.squeeze(gtData["values"])

#Get dimensions
(numTotalFrames, numClasses) = estValues.shape

#Calculate gt and est class based on winner take all
gtClasses = np.argmax(gtValue, axis=1)
estClasses = np.argmax(estValues, axis=1)

numFrames = numTotalFrames / batchSize
costVal = np.zeros((numFrames))
accuracyVal = np.zeros((numFrames))

for iframe in range(numFrames):
    numCorrect = 0
    for ibatch in range(batchSize):
        pvpidx = iframe * batchSize + ibatch;
        costVal[iframe] += -np.log(estValues[pvpidx, gtClasses[pvpidx]])
        if estClasses[pvpidx] == gtClasses[pvpidx]:
            numCorrect += 1
    accuracyVal[iframe] = float(numCorrect)/batchSize


plt.figure()
plt.plot(costVal)
plt.title("Cost vs training")
plt.xlabel("Training time")
plt.ylabel("Cost")
plt.savefig(outDir + "costVsTraining.png")

plt.figure()
plt.plot(accuracyVal)
plt.title("Accuracy vs training")
plt.xlabel("Training time")
plt.ylabel("Accuracy")
plt.savefig(outDir + "accuracyVsTraining.png")




