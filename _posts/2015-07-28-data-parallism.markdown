---
layout: post
title:  "New Data Parallelism in PetaVision"
date:   2015-07-28 23:01:55
author: Sheng Lundquist
categories: jekyll update
---
Hi all,

   An early version of data parallelism is now in the master branch of the Github repository. Here's everything you need to know about the change.

   To run with batches, simply set the parameter *nbatch* in the HyPerCol. This creates a 4th dimension into all layers in the parameter file. Currently, only shared weights are allowed for weights if you are to use batches. These weights are shared across all batches, meaning you have one set of weights that is learning/delivering on all batches.

   While specifying batches is pretty straight forward from a user standpoint, there were many additional changes that were made to incorporate batches.

## Image/Movie Hierarchy:
   The Image/Movie hierarchy has been reworked. Image and Movie now only specifically take images or list of filenames respectively. To read PVP files, use the new ImagePvp and MoviePvp layers. As part of this rework, the parameter to all 4 files for specifying the input file is now *inputPath*. You will need to change your existing parameter files to include this new parameter as a replacement for the old *imagePath*, *listOfFilenames*, etc. Finally, all *start_frame_index* parameters are now 0 indexed, as opposed to the previously 1 indexed values.

   Image and ImagePvp with batches simply copies the image across all batches. Movie/MoviePvp now has a new parameter *batchMethod* that can be one of 3 values: *"byImage"*, *"byMovie"*, and *"bySpecified"* (default). *"bySpecified"* allows you to specify a start frame and skip frame per batch. As an example, if we were to have a parameter file with 4 batches, your start frame and skip frame can be as such:

   ```````````````````````````````````````````````````````
   Movie "myMovie"{
   ...
   inputPath = "myFile.txt";
   start_frame_index = [0, 1, 2, 3];
   skip_frame_index = [4, 4, 4, 4];
   ...
   }
```````````````````````````````````````````````````````

myFile.txt:
```````````````````````````````````````````````````````
0.png
1.png
2.png
3.png
...
```````````````````````````````````````````````````````

Here, batch index 0 will read frames 0, 4, 8 ..., batch index 1 will read 1, 5, 9... etc.

*"byMovie"* and *"byImage"* explicitly sets the *start_frame_index* and *skip_frame_index*. Namely, assuming with a batch size of 4 and the input file having 40 frames:
"byImage":
```````````````````````````````````````````````````````
start_frame_index = [0, 1, 2, 3];
skip_frame_index = [4, 4, 4, 4];
```````````````````````````````````````````````````````

"byMovie":
```````````````````````````````````````````````````````
start_frame_index = [0, 10, 20, 30]
skip_frame_index = [1, 1, 1, 1];
```````````````````````````````````````````````````````


These 2 *batchMethods* also do take a *start_frame_index* (it has to be only one value), which sets an offset into the generated start_frame_index. Examples of how to use the new Image and Movie can be found in the new ImageSystemTest.

## PVP files:
PVP files are completely backwards compatible. Here, we write out each batch as an individual frame. As an example, with 4 batches, the layout of the PVP file will go:

```````````````````````````````````````````````````````
[header]
time = 0
[data for batch 0]
time = 0
[data for batch 1]
time = 0
[data for batch 2]
time = 0
[data for batch 3]
time = 1
[data for batch 0]
etc...
```````````````````````````````````````````````````````

Note that consecutive frames can now have the same timestamp. This will most likely throw errors in your current analysis scripts. The header of PVP files now contains a *nbatch* field (the obsolete *nb* header field) that is written out for use in analysis scripts, as PetaVision itself never reads this parameter. I've added a simple error and sparsity analysis file to *OpenPV/pv-core/mlab/batchLCA* as an example.


Currently, batches cannot be split up into MPI. This is currently my next item on the list. There are also other various reworks internally, but I'll spare you the lengthy explanations, as they're mostly under the hood. Please let me know if you find any bugs or have any problems running with batches.

Sheng

