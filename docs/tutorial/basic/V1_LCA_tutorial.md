# V1 LCA Tutorial

This basic tutorial will walk you through 1) downloading a dataset, 2) unsupervised training of a V1 dictionary using the LCA algorithm, and 3) analyzing the output using one of our automated scripts.  In the next tutorial we will look at how to train an SLP classifier using the V1 dictionary you train here.

The intended audience of this tutorial is able to use some command line tool (eg. Terminal) to build and execute programs, edit text files (eg. vim, emacs), and is familiar with some essential ideas of [artificial neural networks](http://neuralnetworksanddeeplearning.com/chap1.html).  This tutorial guides you in creating a sparse representation (V1) of an image and a subsequent reconstructions using the LCA algorithm described by Christopher Rozell, Don Johnson, Richard Baraniuk, and Bruno Olshausen: [Locally Competitive Algorithms for Sparse Approximations[(http://www.ece.rice.edu/~dhj/rozell_icip2007.pdf).
    
Depending on your internet connection and the speed of your machine you should be able to get to Step 3, "Running PetaVision" and running the experiment in about an hour. Step 3 could take minutes to days depending on the speed of your machine (Many threaded CPUs + GPUs = faster) and how long you choose to train your network.

[TOC]

# 0. Pre-requisites:

1. Successfully build PetaVision and run a BasicSystemTest.  
    - [AWS Installation](https://sourceforge.net/p/petavision/wiki/Install%20PetaVision%20AWS/)
    - [Ubuntu Installation](https://sourceforge.net/p/petavision/wiki/Install%20PetaVision%20Ubuntu/)
    - [OSX Installation](https://sourceforge.net/p/petavision/wiki/Install%20PetaVision%20OSX/)
    
2. Grab a cup of coffee and hunker down. 

# 1. Get your dataset 

You can run this tutorial using any Dataset but since we like the K.I.S.S. principle, we're going to walk through getting a popular and well documented dataset: CIFAR.

CIFAR consists of 50k training images plus 10k test images in 10 categories for classification.  The images are 32x32 color images of the following categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.  The dataset is hosted by the CS department at the University Toronto by Alex Krizhevsky (name sake for award winning 'AlexNet' implementation on Caffe).  This data set is simple to work with (from a whole image classification perspective) and since the images are small you will be able to run your experiments much faster than if you were try using larger pictures. 

For more information about the CIFAR dataset: http://www.cs.toronto.edu/~kriz/cifar.html

## 1.1. Download CIFAR dataset

Either visit the [CIFAR website](http://www.cs.toronto.edu/~kriz/cifar.html) or use the command line to get and unzip the database:
 
    $ cd ~
    $ mkdir dataset
    $ cd dataset
    $ wget "http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz"
    $ tar -zxvf cifar-10-matlab.tar.gz

## 1.2. Extract CIFAR dataset 

You can write your own script to extract and organize the CIFAR images, or you can use the one we're already prepared called 'extractImagesOctave.m'.  The script is located in the PetaVision trunk: PetaVision/mlab/HyPerLCA/extractImagesOctave.m. To use the script, you'll first need to modify extractImagesOctave.m by pointing to the correct local directories. Open the octave script in your favorite text editor and following the instructions.

    $ cd ~/workspace/PetaVision/mlab/HyPerLCA
    $ vim extractImagesOctave.m

Make sure you saved your changes to the script before continuing to the next step. 

## 1.3.  Extract images using octave script
    
Navigate to where you unzipped the cifar-10-matlab.tar.gz file and extract the images in octave:

    $ cd ~/dataset/cifar-10-batches-mat/
    $ octave

    > addpath('path/to/PetaVision/mlab/HyPerLCA')
    > extractImagesOctave('data_batch_1.mat',1)
    > extractImagesOctave('data_batch_2.mat',2)
    > extractImagesOctave('data_batch_3.mat',3)
    > extractImagesOctave('data_batch_4.mat',4)
    > extractImagesOctave('data_batch_5.mat',5)
    > extractImagesOctave('test_batch.mat',0)

## 1.4. Combine the data_batches to make a master file
    
Each run of extractImagesOctave produced a unique text file listing all the images in random order called 'randorder.txt'.  If you wish to expand your training dataset to  include all of the training images, you can concatenate them by copying them to a common directory with different names (to avoid clobbering the files) and then doing:    

    $ cat *.txt > mixed_cifar.txt

Congratulations!  You now have a massive training dataset along with a test set that you will use in the next tutorial in creating a classifier.  For now we are only concerned about using the dataset for unsupervised learning.
    
# 2. The Params File
    
The params file is where each experiment is described in English for PetaVision. It details the different objects (ie. column, layers, and connections) that are used by PetaVision.
    
You'll be starting off with a params file that has already been tuned pretty well but feel free to modify parameters as you experiment to try to identify different results.  

The following picture is a simplified graphical representation of the params file for this run.

![V1-LCA](https://sourceforge.net/p/petavision/code/HEAD/tree/trunk/docs/tutorial/basic/V1_LCA_simple.png?format=raw)


If you are using the PetaVision Public AMI, you can make these drawings of any params file just by typing:

    $ draw [name of params file]
    
## 2.1. Find your params file: V1_LCA.params    
    
Go to where you downloaded PetaVision and navigate to: /PetaVision/docs/tutorial/basic/
In the case of AWS, you may want to copy the params file to your EBS volume in the event that your instance gets outbid and shut down. 

## 2.2. Inspect the params file
    
First just look over the params and see if you can understand the general structure of a params file.  It is organized into three categories: the 1.column, 2.layers, and 3.connections, to simulate a cortical column in the brain. The params file you will be using is commented to help guide you along to highlight this structure of the params file. 

### 2.2.1. HyPerCol

The column is what holds the whole experiment and all the layers are proportional to the column.  In PetaVision the column object is called HyPerCol for 'High Performance Column'. The 'y' is there because it spells hyper (a.k.a: legacy naming scheme stuff)

The column sets up a bunch of key experiment details such as how long to run, where to save files, how frequently and where to checkpoint, and adaptive time-step parameters. All of these parameters are fairly clearly identified but lets look at a few of the very important ones:

HyPerCol Parameter  |  Description
--------------------|--------------------------------------------------------------------
startTime           |  sets where experiment starts; usually 0
stopTime            |  sets how long to run experiment; (start - stop)/dt = number of timesteps
dt                  |  how long a timestep; modulations possible with adaptive timestep 
outputPath          |  sets directory path for experiment output
nx                  |  x-dimensions of column; typically match to input image size
ny                  |  y-dimensions of column; typically match to input image size
checkpointWriteDir  |  sets directory path for experiment checkpoints; usually output/Checkpoints
dtAdaptFlag         |  tells PetaVision to use the adaptive timestep parameters for normalized error layers

For more details on the HyPerCol please read the documentation:[HyPerCol Parameters](http://petavision.sourceforge.net/doxygen/html/classPV_1_1HyPerCol.html#member-group)

### 2.2.2. HyPerLayer

The layers are where the neurons are contained and their dynamics described. You can set up a layers that convolve inputs, have self-self interactions, or even just copy the layer properties or activities of one layer to another ... and more. All layers are subclassed from HyPerLayer and you can read about their individual properties by following some of the doxygen documentation.

Some important parameters to notice are nxScale, nyScale and nf since they set up physical dimensions of the layer. phase and displayPeriod describe some of the temporal dynamics of the layer.  Most layers have their own unique properties that you can explore further on your own.  For now this is a good snapshot. The table below summarizes the types of layers we use and roles in this experiment:
        
Layer Class             |  "Name"  |  Description
------------------------|----------|------------------------------------------------
Movie                   | "Image"  | loads image from imageListPath
ANNNormalizedErrorLayer | "Error"  | computes residual error between Image and V1
HyPerLCALayer           | "V1"     | makes a sparse representation of Image using LCA
ANNLayer                | "Recon"  | output for visualization

Before moving on to Connections, we should make a note about displayPeriod, writeStep,  triggerFlag, and phase. Movie has a parameter 'displayPeriod' that sets the number of  timesteps an image is shown. We then typically set the writeStep and initialWriteTime to be some integer interval of displayPeriod, but this isn't necessary. For example if you want to see what the sparse reconstruction looks like while the same image is being shown to Movie, you can change the writeStep for "Recon" to 1 (just note that your output file will get very large very quickly so you may want change the stopTime to a smaller value if you want this sort of visualization).

While writeStep has to do with how frequently PetaVision outputs to the .pvp file (this is the unique binary format used for PetaVision), the triggerFlag in more in with the dynamics of the layers.  Notice only the "Recon" layer has a trigger flag and that the triggerLayerName = "Image".  This means that PetaVision will only process the convolution of the "Recon" after a new image is shown.  

But don't we want it to make the convolution using the sparse representation found at the end of the displayPeriod?  Keen observation. This is where phase comes in. Phase determines the order of layers to update at a given timestep.  To get the Recon from V1 before the new image makes its way to V1 and starts changing the sparse representation, we set phases as follows: 

Layer Class             |  "Name"    |  Phase
------------------------|------------|------------
Movie                   | "Image"    |    0
ANNNormalizedErrorLayer | "Error"    |    1
HyPerLCALayer           | "V1"       |    2
ANNLayer                | "Recon"    |    1



**********
*********       Absolute Value of the Error Layer
*******

For more details on the HyPerLayer parameters please read the documentation:[HyPerLayer Parameters](http://petavision.sourceforge.net/doxygen/html/classPV_1_1HyPerLayer.html#member-group)

        
### 2.2.3. Connections
        
The connections connect neurons to other neurons in different layers. Similar to layers, connections are all subclassed from their base class HyPerConn. Connections are where the 'learning' of an artificial neural network happens.

Connections in PetaVision are always described in terms of their pre and postLayerName, their channel code, and their patch size (or receptive field). Some connection parameters are inherited from another connection such as patch size for a TransposeConn or CloneKernelConn from the originalConnName.  We use a naming convention of [PreLayerName]To[PostLayerName] but it is not required if you explictly define the pre and post layer. 
        
The channelCode value determines if the connection is excitatory (0), inhibitory (1), or neither (-1).  Neither is useful when you are making a connection to an error layer to train the weights, but want the activity for the layer to come from a reconstruction layer. 

Patch size is determined by the nxp, nyp, and nfp parameters.  Restrictions on how you can set these values are explained in detail in [Patch Size and Margin Width Requirements].
        
The following table summarizes the types of connections that are used and their
roles in this experiment:

Connection Class  |  "Name"        |   Description
------------------|----------------|-------------------------------------------
HyPerConn         | "ImageToError" | loads image from imageListPath
MomentumConn      | "V1ToError"    | computes residual error between Image and V1
TransposeConn     | "ErrorToV1"    | makes a sparse representation of Image using LCA
CloneKernelConn   | "V1ToRecon"    | clone V1ToError and convolve with V1 to make a reconstruction

For more details on the HyPerConn parameters please read the documentation: [HyPerConn Parameters](http://petavision.sourceforge.net/doxygen/html/classPV_1_1HyPerConn.html#member-group)

## 2.3. Customize the params file for a run on your system

The params file is tagged to let you know where you have to edit parameters before you run. The parameter will have a ! symbol at the beginning of the line if you need to edit it. If a parameter is tagged, there will be a small commented instruction following the tag. Before you move on to running the experiment, make sure you delete every !. When done, save the file and you will be ready to start your run.  The sections below identify  objects to make sure to review, however there are some extra ! comments you will want to look for (eg. writeStep is commented in all the layers since you may want to adjust depending on how frequently you plan to 

* NOTE: Directory paths assume you organized your workspace as: ~/workspace/PetaVision
If this not where you put your PetaVision build, take note of where it is and update the file paths accordingly as you move through the tutorial.

### 2.3.1. HyPerCol "column"

Parameter                   |       What to do
----------------------------|------------------------------------------------------------
stopTime                    | Currently at 10,000,000 = 1 time through the data set; multiply to run through dataset more times
outputPath                  | Change to where you want to save your output
checkpointWriteDir          | Change to where you want to save your checkpoints (usually outputPath/Checkpoints)
        
### 2.3.2. Movie "Image" | Update Image path 

Parameter       |       What to do
----------------|------------------------------------------------------------
imageListPath   | Change to point to your mixed_cifar.txt file created in step 1

        
### 2.3.4. [Optional] MomentumConn "V1ToError" | Load weights from file]
        
This tutorial is designed to bring you from chaotic random weights to beautiful colorful gabors.  However, if you don't want to wait for your dictionary to mature and want to start off with well trained weights, we have included a well trained dictionary located at:
            
Parameter       |       What to do
----------------|------------------------------------------------------------
weightInitType  | Uncomment the one ending with "FileWeight";
initWeightsFile | Change to point to ~/workspace/PetaVision/docs/tutorial/V1_LCA/V1ToError_W.pvp

Delete or comment out with two slashes // the following paramaters

* weightInitType  Comment the one ending with "UniformRandomWeight";
* wMinInit
* wMaxInit
* sparseFraction

# 3. Running PetaVision

You've arrived at the moment of truth.  These final steps should take a couple of minutes if everything else has been working smoothly.  PetaVision has a built in params file error checker and will halt the run if something was misspelled.  You'll just want to pay attention to the runtime output to decipher where you have to fix the params file.  Frequently the error is a missing semi-colon ';' at the end of a line or a misspelled parameter.  PetaVision specific errors (eg. incompatible patch sizes or layers) produced unique error messages with detailed instructions about how to try to fix our params file. 

## 3.1 Build PetaVision

If you haven't already built PetaVision make PetaVision now.  You can follow the instructions from any of the installation instructions found in the doxygen documentation: http://petavision.sourceforge.net/doxygen/html/md_src_install_aws.html

You can do everything from BasicSystemsTest or if you prefer you can check out one of the sandboxes (eg. HyPerHLCA).  Just make sure you add the path to the sandbox to the CMakeLists.txt in the parent directory or PetaVision won't be able to build the executable file (the sandboxes are commented out at the bottom of the file, just scroll down and uncomment them to move on).

## 3.2 Start a Screen

Since you'll probably want to be able to use your computer while you are running your
experiment, we recommend using 'screen'

    screen -S run
    
To detach the screen: 'control+a+d'
To reattach the screen: 'screen -r run'

## 3.3 Run-time arguments

Make sure you are attached to your 'run' screen. I like to navigate to the location 

Run-time flag            |       Description
-------------------------|--------------------------------------------------------------
-p [/path/to/pv.params]  | Point PetaVision to your desired params file
-t [number]              | Declare number of CPU threads PetaVision should use
-c [/path/to/Checkpoint] | Load weights and activities from the Checkpoint folder listed
-d [number]              | Declare which GPU to use at an index; not essential
-l [/path/to/log.txt]    | PetaVision will write out a log file to the path listed


All of these flags combined you'll get a runtime argument that looks similar to this:

    ~/workspace/HyPerHLCA/Release/HyPerHLCA -p ~/workspace/input/params/V1_LCA.params -t 8 -l ~/workspace/output/txt.log
    
# 4. Analyze Run

PetaVision has tools to review the run in progress and after the experiment is finished.  You will either be looking at files in the outputPath directory or in one of the Checkpoint directories.  This params file and the corresponding analysis tool kit is set up to have you look at the files in the outputPath directory.

The main type of file you'll be examining are the '.pvp' files.  This is a PetaVision specific binary file type that saves space, can be read using python or matlab/octave, and can easily be loaded into PetaVision.  

## 4.1 outputPath directory files


In your output directory you should see the following files:

File or Folder/         |   Description
------------------------|-----------------------------------------------------------------
a0_Image.pvp            | Image layer activity written on Image writeStep frequency
a1_Error.pvp            | Error layer activity written on Error writeStep frequency
a2_V1.pvp               | V1 layer activity written on V1 writeStep frequency
a3_Recon.pvp            | Recon layer activity written on Recon writeStep frequency
Checkpoints/            | Contains folders of Checkpoints written on checkpointWriteStepInterval
Error_timescales.txt    | Log of Error layer timescales
HyPerCol_timescales.txt | Log of aggregated Error timescales; adaptive time-step information
log.txt                 | Generated from -l flag; saved stdout 
pv.params               | PetaVision generated params file; removes all the comments; preferred for drawing diagrams
timestamps/             |
w1_V1ToError.pvp        | Weight values written on writeStep frequency

Depending on how long you ran your experiment for and how frequently you set writeStep, the size of your .pvp files can range from kB to gB.  

If you are on the AWS PetaVision Public AMI, we have already installed our params file drawer.  If you want to test this out, type:

    $ draw pv.params

## 4.2 Checkpoint directory files

Navigate to one of the checkpoints and you'll see there are many more files that are saved than in the outputPath directory.  This is because a Checkpoint includes all the information PetaVision would need to initialize a run including timers, layer potentials and activities, weights, and more.  

Without going into all of the files, one important file to notice is the V1ToError_W.pvp file. This is the file we will use in the next tutorial for classification.  This file is a snapshot of the weights (dictionary elements) that 

## 4.3 Run automated analysis script

You can write your own analysis scripts and should look at the one we are about to use for reference if you want to do that, but for now let's just use it as is.  The script we are working with is called 'analyze_network.m'.  All we have to do is navigate to the outputPath directory and type the following commmand:

    octave ~/path/to/PetaVision/mlab/util/analyze_network.m

This script looks in your current directory for pv.params to find the names and properties of the different will analyze the the .pvp files in the root output directory and produce graphical
outputs in the newly created folder 'Analysis'

## 4.4 [Cloud - AWS] - view the files on your local machine

One extra step for you AWS users: scp the files from the AWS instance to your local machine to view.

    $ scp -r -i ~/.ssh/cred ec2-user@[000.000.000.000]:/home/ec2-user/mountData/V1_LCA/output/Analysis .

# 5. Experiment

Now is your chance to explore and experiment some with the different parameters.  Maybe you want to reduce the displayPeriod, modify the threshold, or change the learning rate of your connections.  Perhaps you want to use a totally different dataset.  

Whatever you do, be sure to come back and tune in when we use the weights that you just trained to design a SLP classifier. 

# 6. Comments / Questions?

I hope you found this tutorial helpful. If you identify any errors and opportunities for improvement to this tutorial, please contact the developers of PetaVision using the e-mail listed on sourceforge with "V1_LCA_tutorial" in the subject line.

