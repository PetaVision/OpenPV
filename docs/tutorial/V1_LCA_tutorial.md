# V1 LCA Tutorial

This basic tutorial is set up to walk you through downloading a dataset, performing
unsupervised learning on a V1 dictionary of that dataset using the LCA algorithm, and
finally looking at your output using one of our automated scripts.  In the next tutorial
we will look at how to train an SLP classifier using your V1 dictionary.

0. Pre-requisites:

    1. Successfully installed PetaVision and run a BasicSystemTest
    2. Grab a cup of coffee

1. Get your dataset (we'll describe how to get the CIFAR dataset)

    1. Download CIFAR dataset: http://www.cs.toronto.edu/~kriz/cifar.html
    
        If you are on the AWS server or on your local machine you can use wget

        $ cd ~
        $ mkdir dataset
        $ cd dataset
        $ wget "http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz"
        $ tar -zxvf cifar-10-matlab.tar.gz
        
    3. Extract CIFAR dataset using PetaVision/mlab/HyPerLCA/extractImagesOctave.m
    
        You'll need to first modify extractImagesOctave.m
        Follow the instructions at the top of the script
        
        $ cd ~/path/to/PetaVision/mlab/HyPerLCA
        $ vim extractImagesOctave.m
        
        Make sure you saved your changes to the script before continuing
        
    3. Navigate to where you unzipped the cifar-10-matlab.tar.gz file and extract images
    
        $ cd ~/dataset/cifar-10-batches-mat/
        $ octave
        
        > addpath('path/to/PetaVision/mlab/HyPerLCA')
        > extractImagesOctave('data_batch_1.mat',1)
        > extractImagesOctave('data_batch_2.mat',2)
        > extractImagesOctave('data_batch_3.mat',3)
        > extractImagesOctave('data_batch_4.mat',4)
        > extractImagesOctave('data_batch_5.mat',5)
        > extractImagesOctave('test_batch.mat',0)
        
        * Note: I recognize this is not an elegant method, but it works and is clear
        
    4. Combine the data_batches to make a master file
    
        Each run of extractImagesOctave produced a unique text file listing all the
        images in random order.  If you wish to expand your training dataset to 
        include all of the training images, you can concatenate them by copying them 
        to a common directory with different names (inelegant solution) and then doing:
    
        > cat *.txt > mixed_cifar.txt
        
    Congratulations!  You now have a massive training dataset along with a test set
    that you will use in the next tutorial in creating a classifier.  For now we
    are only concerned about using the dataset for unsupervised learning.
    
2. Fix up your params file
    
    The params file is where each experiment is described in english for PetaVision.
    It details the different objects (ie. hypercol, layers, and connections) that 
    are used by PetaVision.
    You'll be starting off with a params file that has already been tuned pretty
    well but feel free to modify parameters as you experiment to try to identify 
    different results.  
    
    1. Get your params file: /PetaVision/docs/tutorial/basic/V1_LCA.params    
    
        In the directory you will also see a V1_LCA.png that has a graphical 
        rendition of this params file.  If you are on AWS
        Copy this file to a directory you will be working from.  
        In the case of AWS, you may want to copy this file to your EBS volume
        
    2. 

3. Run PetaVision

4. Analyze Results

5. Experiment
        
    