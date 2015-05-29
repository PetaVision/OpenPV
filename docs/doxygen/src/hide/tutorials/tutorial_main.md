# PetaVision Tutorial

## Introduction
The following tutorials will take you through installing and using the fundamental features of PetaVision


 1. Download, Build & Run SystemTest
    -  [AWS Installation](../tutorial/install/install_aws.md)
    -  [OS X Installation](../tutorial/install/install_osx.md)
    -  [Ubuntu Installation](../tutorial/install/install_ubuntu.md)
    -  [Ubuntu Installation](/../tutorial/install/install_ubuntu.md)
    -  [Ubuntu Installation](install_ubuntu.md)
    
 2. \subpage mermaid "Visualize your params file (only OS X)"
 
 3. \subpage v1lca "Train and Visualize a V1 Dictionary"
 
 4. \subpage classification "Whole Image Classification"


\page mermaid Visualize The Params File

Visualization is one of the fastest and easiest ways to determine if your params file is set-up as you intended and has become a popular debugging tool for PetaVision.  In this tutorial, you will install Mermaid and use it to generate a flow diagram of your params file.

Install Mermaid
-----------------------------------
1) For OS X, download the ‘node’ .pkg file from https://nodejs.org/download/; this will install the npm package manager.  If you are using a package manager, please refer to this link on [how to install nodejs](https://github.com/joyent/node/wiki/Installing-Node.js-via-package-manager
): 

2) After you install nodejs, in your command line install mermaid and phantomjs:

    > sudo npm install -g mermaid
    > sudo npm install -g phantomjs
    
3) Mermaid will interpret plain text files and output .png files. To test your install, make a test document with the following text (ignoring quotes):

	“graph TB;
	 a -—> b; “

4) Enter the following commands:

	> mermaid your_test_file
	> open your_test_file.png

5) This should create and display a png with the same name as your text file, displaying a simple flowchart. 

DRAWING A DIAGRAM
-----------------------------------

Make a copy of the params file you would like to draw and place it in the same directory as the python script “integrated_param_extractor.py”. In a default install of a sandbox or the PetaVision trunk, this script can be found in PetaVision/plab.

In this directory, enter the following commands:

	> python integrated_param_extractory.py ~/workspace/Petavision
	> mermaid param_graph
	> open param_graph.png

(As with the test suggested above, the first command creates an input file named “param_graph” for the mermaid CLI, the second creates a .png file from it)





\page v1lca Train and Visualize a V1 Dictionary

#\page classification Whole Image Classification

<img src="neural-fire.png" alt="Drawing" style="width: 800px;"/>

At this point, you have already installed PetaVision either on your local computer or available server and have been able to both run the systems tests (Lesson 0) and train a dictionary on a basic LCA set-up (Lesson 1).  Now we arrive at Classification, one of the standard machine learning and computer vision tasks of the day.

In general, two tasks are being performed when you are aiming to classify an image: Step 1) generate a sparse representation of the original image [time-consuming], Step 2) aim that sparse representation at a classifier and guess the object category [fast].  For each of these tasks, a separate set of dictionaries are needed.  

For this lesson, we will start with a dictionaries trained on the 20 PASCAL categories to save you the time, however, if you want to do the exercise from scratch with any categories you want instructions are included.  For this tutorial, we’ll first use the pre-trained dictionaries.  You will just need to load your weights from file by slightly modifying the params file.  Also, you’ll want to have images from the PASCAL categories in x-y dimensions of 192 x 256.

This lesson can take ~1-3 hours depending on how many images you attempt to classify, if you separate step 1 and step 2, and the power of your computer. 


Getting your images
--------------------------------
We provide a sample set of images, however, if you obtain images that include one of the 20 object classes from the PASCAL VOC dataset in the following dimensions: x = 192, y = 256, you should be able to perform localized image classification using PetaVision.

The 20 PASCAL VOC categories are common objects:

1. aeroplane
2. bicycle
3. bird
4. boat
5. bottle
6. bus
7. car
8. cat
9. chair
10. cow
11. dining table
12. dog
13. horse
14. monitor
15. motorbike
16. person
17. potted plant
18. sheep
19. sofa
20. train

For the original dataset available here [note website is occasionally down], images were pulled from flickr and cover a range of lighting, orientations, occlusions, truncations and are annotated.  These annotations are essential for training the classifier dictionary (step 2) by providing a ground truth.  As we learned in Lesson 1, it is not necessary to have a ground truth to perform unsupervised learning of a dictionary on natural images, which is the dictionary that we will be using for step 1.

If you are making your own set of object classification, you’ll want to check out Lesson 3 - Training A Classifier.  For now, let’s just play with the PASCAL categories.  
I’m just going to use the images you give me
Download these sample images [images.zip]
I want to use my own images
If you are using your own images, make sure you get them into the correct dimensions using cropping or equal-ratio image shrinking (maintain the original xy ratios of the image to increase chances of getting an accurate classification). 

If you downloaded the sample images, you only need to modify img_path.txt to match the directory path to your images or you can make your own.  If you just want to modify the existing file, open img_path.txt in your favorite text editor and change the path to navigate to the image files from where you have them unzipped and saved. 

If you are using your own images or just want to make your own text list, use the following bash command in the unzipped directory with the images: 

    ls -1d $PWD/*.png > img_path.txt

Open the file and confirm img_path.txt looks good.  All done. 
Modifying the params file
Fortunately, you have a fancy working params file to start with.  You will only need to change a couple of lines of the params file to point to the correct paths where you are storing your files. 
Output directory
In the HyPerCol change the outputPath parameter to match where you want PetaVision to output your files.  
Input files
In Movie layer “Image”, change imageListPath to the location where you saved imageListPath = “path/to/img_path.txt”;

Now scroll down to the CONNECTIONS and for every connection that has wegithInitType = “FileWight” change the initWeightsFile paramater to the path for the corresponding HyPerConn connections that you got with the files for this tutorial:
    S1ToImageReconS1Error_W.pvp
    S1MaxPooledToGroundTruthReconS1Error_16X12_W.pvp
    S1SumPooledToGroundTruthReconS1Error_16X12_W.pvp
    BiasS1ToGroundTruthReconS1Error_16X12_W.pvp
    S1MaxPooledToGroundTruthReconS1Error_4X3_W.pvp
    S1SumPooledToGroundTruthReconS1Error_4X3_W.pvp
    BiasS1ToGroundTruthReconS1Error_4X3_W.pvp


Running PetaVision
--------------------------------------
Since the run should take some time, it is best that you make use of a computer with some CPU and GPU power. 

1. use screen
    ~~~~~~~~~~~~~~~~~~
    screen -S run
    ~/workspace/HyPerCol/Release/HyPerCol -p sparseClassify.params -t 24 2&>1 
    ~~~~~~~~~~~~~~~~~~
2. exit screen using:
    ~~~~~~~~~~~~~~~~~
    ctrl+a+d
    ~~~~~~~~~~~~~~~~~
2. use top to confirm you are using the CPUs
3. nvidia-smi to check GPU status
4. less path/to/output/directory/HyPerCol_timescales.txt
	ESC  +  > to update
5. go for a walk


##Analyzing Results
analyze_network.m is the script you will want to run to analyze your run and the script is found in PetaVision/mlab/util. Before running the script you will want to edit your ~/.octaverc file with the following line modified to point to your PetaVision directory:

    addpath(‘/path/to/your/PetaVision/mlab/util’)
	
First, navigate to your output directory then from the output directory type the command:

    octave path/to/your/PetaVision/mlab/util/analyze_network.m

This script will output a bunch of files to a folder called Analysis in your output directory. Open the folder and let’s start looking at the results.

The current params file only looks at the first two images but the follow table lists all the images if you want to show more by increasing stopTime in HyPerCol to 2001 instead of 401.


##Conclusions and Remarks

Percent estimation of each category in each image (softmax)
Please add to PetaVision:
Scale and rotational invariance
Scanning across image at different resolutions

###Add to Tutorial
Explain why we have two resolutions of ground truths?
How do the sum pooling and max pooling layers work?
Explain what are the strengths and limitations of this classification system?  Propose some ways of improving the system.


 
