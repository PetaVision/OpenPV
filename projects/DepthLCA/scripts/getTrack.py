from parseTrackletXML import parseXML
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import numpy as np

dataDir = "/nh/compneuro/Data/KITTI/2011_09_26/2011_09_26_drive_0001_sync/"
trackFile = dataDir + "tracklet_labels.xml"
imageDir = dataDir + "image_02/data/"

for trackletObj in parseXML(trackFile):
    [h, w, l] = trackletObj.size
    for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in trackletObj:


        imageFile = imageDir + str(absoluteFrameNumber).rjust(10, '0') + ".png"
        img = mpimg.imread(imageFile)
        [x, y, z] = translation
        #Draw box
        img[y:y+w,x,0] = 1
        img[y:y+w,x,1] = 0
        img[y:y+w,x,2] = 0
        img[y:y+w,x+h,0] = 1
        img[y:y+w,x+h,1] = 0
        img[y:y+w,x+h,2] = 0
        img[y,x:x+h,0] = 1
        img[y,x:x+h,1] = 0
        img[y,x:x+h,2] = 0
        img[y+w,x:x+h,0] = 1
        img[y+w,x:x+h,1] = 0
        img[y+w,x:x+h,2] = 0

        plt.imshow(img)
        plt.show()

    #end: for all frames
#end: for all tracklets


