---
layout: post
title:  "Image from Memory Buffer"
date:   2014-11-7 23:01:55
author: Pete Schultz
categories: jekyll update
---

I have a subclass of image in the ImageFromMemoryBuffer sandbox that does what I think Gar means by server mode.  You can do the following in a pipeline:

- create a HyPerCol object where one of the layers is an ImageFromMemoryBuffer object
- call the ImageFromMemoryBuffer object's setMemoryBuffer method to set the image, offsets, and offset anchor
- call the HyPerCol's run method
- call setMemoryBuffer method again with the next image
- call the HyPerCol's run method again with a new stopTime
- etc.

The limitations so far are:

- It doesn't handle autoRescale yet.  (Does Image handle autoRescale if it's given a .pvp file?  I think it doesn't, but I haven't checked.  If it does that could be added to autoRescale easily.  If not, we should add it to Image in a way that it can be both for reading images from disk or receiving it as a memory buffer.

- The layerLoc needs to be the same for all the images, because the HyPerCol's nx and ny and the image's nxScale and nyScale can't change.  A work-around for this would be to create a new HyPerCol for each image (or at least each time the desired layerLoc size changed) but that could get expensive, since sometimes the initialization of the layers and connections takes a while.  Should we discuss how to make the image size more flexible from frame to frame?  Or, since we're interested in classification, detection, and localization, should we make do with rescaling the input image, and allowing (if the aspect ratio changes) for the fact that the images might come in looking letterboxed horizontally or vertically?

Also, we've discussed this before, but it would be nice if Movie could receive a video format file as input and extract frames on the fly.  If so, would it make sense for server mode to receive the video as a whole as opposed to receiving the frames one at a time?

