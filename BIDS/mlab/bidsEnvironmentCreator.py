import matplotlib.pyplot as plt
import numpy as np
import random
import os
from math import floor
from scipy.misc import imsave

#Params
#Node radius is how big to make the dot.
#Radius of 1 means it's expanded to a square that's 3 by 3, with orig dot in center
#Note: there is a change for nodes to overlap in the image
nodeRadius = 1

jitterVal = 9
worldSize = (256, 256) #y by x
bidsSize = (20, 20) #y by x, number of bids nodes in world
outputDir = "./envFigs/"
filename = "envFig_"
ext = ".png"

def makeFig(jitterVal, worldSize, bidsSize, nodeRadius, outputFilename):
    world = np.zeros(worldSize);
    spacing = (float(worldSize[0])/bidsSize[0], float(worldSize[1])/bidsSize[1])

    #Make sure radius isn't going to make node go off screen without jitter
    if floor(float(spacing[0])/2) <= nodeRadius:
        print "Radius too big, max val", floor(float(spacing[0]/2)) - 1
        return
    #Start spacing halfway spacing
    nodes = (list(np.arange(float(spacing[0])/2, worldSize[0], spacing[0])),
             list(np.arange(float(spacing[1])/2, worldSize[1], spacing[1])))

    #Jitter
    jnode = [(floor(y+random.randint(-1 * jitterVal, jitterVal)), floor(x+random.randint(-1 * jitterVal, jitterVal))) for y in nodes[0] for x in nodes[1]]
    #Make sure that boundary conditions consider node radius as well.
    #A node will only be shown if the node in it's entirety is in bounds
    jnode = [(y, x) for (y, x) in jnode if y - nodeRadius>=0 and y + nodeRadius<worldSize[0] and x - nodeRadius>=0 and x + nodeRadius<worldSize[1]]

    #if JITTER:
    #    jnode = [(floor(y+random.randint(-1 * jitterVal, jitterVal)), floor(x+random.randint(-1 * jitterVal, jitterVal))) for y in nodes[0] for x in nodes[1]]
    #    jnode = [(y, x) for (y, x) in jnode if y>=0 and y<worldSize[0] and x>=0 and x<worldSize[1]]
    #else:
    #    jnode = [(floor(y), floor(x)) for y in nodes[0] for x in nodes[1]]


    for coor in jnode:
        for curRadius in range(nodeRadius + 1):
            #Square
            for horL in range(int(coor[1]-curRadius), int(coor[1]+curRadius+1)):
                #Top line
                world[coor[0]-curRadius][horL] = 1
                #Bot line
                world[coor[0]+curRadius][horL] = 1
            for vertL in range(int(coor[0]-curRadius), int(coor[0]+curRadius+1)):
                #Left line
                world[vertL][coor[1]-curRadius] = 1
                #Right line
                world[vertL][coor[1]+curRadius] = 1


    #save img
    imsave(outputFilename, world)


if not os.path.exists(outputDir):
    os.makedirs(outputDir)

#No jitter
makeFig(0, worldSize, bidsSize, nodeRadius, outputDir + filename + "nj" + ext)
#jitter
makeFig(jitterVal, worldSize, bidsSize, nodeRadius, outputDir + filename + "j" + ext)
