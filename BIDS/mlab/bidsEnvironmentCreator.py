import matplotlib.pyplot as plt
import numpy as np
import random
import os
from math import floor
from scipy.misc import imsave
import Image, ImageDraw

#Params
#Node radius is how big to make the dot.
#Radius of 1 means it's expanded to a square that's 3 by 3, with orig dot in center
#Note: there is a change for nodes to overlap in the image
nodeRadius = 1

#Using 256 by 256, with 64 by 64 nodes, jitter 3
worldSize = (400, 400) #y by x
bidsSize = (56, 56) #y by x, number of bids nodes in world
borderSize = 20 #Number of pixels between 2 images
cBorderRatio = 2 #vertical center is 3 times as big as border size
jitterVal = 6
outputDir = "./envFigs/"
filename = "envFig"
ext = ".png"

def makeFig(jitterVal, worldSize, bidsSize, nodeRadius):
    world = np.ones(worldSize);
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
                world[coor[0]-curRadius][horL] = 0
                #Bot line
                world[coor[0]+curRadius][horL] = 0
            for vertL in range(int(coor[0]-curRadius), int(coor[0]+curRadius+1)):
                #Left line
                world[vertL][coor[1]-curRadius] = 0
                #Right line
                world[vertL][coor[1]+curRadius] = 0

    return world
    #save img
    #imsave(outputFilename, world)


if not os.path.exists(outputDir):
    os.makedirs(outputDir)

#No jitter
njWorld = makeFig(0, worldSize, bidsSize, nodeRadius)
#jitter
jWorld = makeFig(jitterVal, worldSize, bidsSize, nodeRadius)
vertBorder = np.zeros((worldSize[0], borderSize))
cVertBorder = np.zeros((worldSize[0], borderSize*cBorderRatio))
horBorder = np.zeros((borderSize, worldSize[1] * 2 + borderSize * 2 + borderSize * cBorderRatio))
world = np.concatenate([vertBorder, njWorld, cVertBorder, jWorld, vertBorder], axis=1)
world = np.concatenate([horBorder, world, horBorder], axis=0)
outputFilename = outputDir + filename + ext
imsave(outputFilename, world)


#Read image back in using image
im = Image.open(outputFilename)
draw = ImageDraw.Draw(im)
#Draw location is x by y
#left image
#upper left
draw.text((10, 10), "0", 255)
#lower left
draw.text((2, worldSize[0]+borderSize - 10), "255", 255)
draw.text((20, worldSize[0]+borderSize), "0", 255)
#lower right
draw.text((worldSize[1]+borderSize - 10, worldSize[0]+borderSize), "255", 255)

#right image
#upper left
draw.text((worldSize[1]+borderSize*(cBorderRatio) + 10, 10), "0", 255)
#lower left
draw.text((worldSize[1]+borderSize*(cBorderRatio) + 2, worldSize[0]+borderSize-10), "255", 255)
draw.text((worldSize[1]+borderSize*(cBorderRatio) + 20, worldSize[0]+borderSize), "0", 255)
#lower right
draw.text((2*worldSize[1]+borderSize*(cBorderRatio+1) - 10, worldSize[0]+borderSize), "255", 255)

del draw
im.save(outputFilename, "PNG")
