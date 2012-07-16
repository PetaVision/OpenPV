BEGIN README.TXT


ImageNet database management scripts. 
Written by:
      Dylan Paiton
      Los Alamos National Laboratory, Group ISR-2 
      paiton@lanl.gov, dpaiton@gmail.com

    Please contact me with any errors you may have

If you wish to learn more about a script: read this document, read its comments, then contact me.
None of these scripts are completely finished. They mostly have TODO lists in them, and the ones that don't probably should.

Most (except for I, V, VI, VII) of these scripts are currently set up to be called from within other programs.
You must edit them (it is specified how near the beginning of each program) in order to run them from a command prompt.


I] To download new images:
    ./downloadImages.pl search_terms.txt

    This script requires a user name and access-key, given by Image-Net.
    I have one for downloading at LANL and the NMC, if you need it contact me.
    The program is NOT to be distributed unless a different user/key is used.

    This program will download the tar image archives for each term specified by search_terms.txt
    The files are downlaoded to the current_directory/../img/ directory into folders named according to the search term.
    The program will modify search_terms.txt to comment out completed downloads.

    NOTE: In order to use this script on LANL computers, you must set the $use_proxy flag in the script to '1'. 

    First prompt:
        Each Synset (category) on Image-Net may contain a list of sub-categories.
        If you chooose to download all child synsets, you will download all sub-categories of your given search term.
        Be careful with this, there can be a LOT of sub-categories.
        The sub-categories will be placed into appropriately-named folders.
        The folder structure will be flat with respect to the hierarchy, with the parent folder name being your original search term.
    
    Second prompt:
        If you choose no (n) for the first prompt, you will be asked how many images you wish to download.
        This number is calculated by using the average of the number of images per tar archive downloaded.
        The number is based on a small (~50) subset of archives, and therefore is not extremely accurate.
        I attempt to over-estimate the amount of images downloaded, so that you at least get the ammount you request.


    I.i] Search terms (in the file search_terms.txt):
        Terms should be separated by a return only. Each line will represent a new search.
        
        Please check your search terms on image-net.org to verify that you are going to get the result you expect.
        The script downloads the 'most popular' synset, as defined by Image-Net.
        You see this by searching for a term and looking at the Popularity percentile.
        The top result is the most popular by default.

        The program allows for comments in the input file. Any line preceded with a # will be ignored.


II] To count images:
    ./countimages.pl "num_categories" "category"

    This will scan current_directory/../img/ for .tar archive files.
    The program will count the number of files in each archive it fount.

    Inputs:
    The program will return the top (most populated) "num_categories" categories as an array.
    You may search a specific category (i.e. "dog, domestic dog, Canis familiaris") by using this input.
    If you pass "rt" as the "category" then the program will count all of the parent categories.

    NOTE: A tar error will occur if you are actively downloading while running this script. It does not interrupt the script.


III] To get a list of a category's parents within the Word-Net hierarchy:
    ./findParents.pl "category" ("previous-parent-array")

    This will return the parents of a category according to the current word-net hierarchy posted on image-net.org
    This program often asks the user to choose between many parental trees. This is becasue the image-net hierarchy contains categories in many different loations.

    The optional parameter is used when finding the parents of many different categories.
    If you pass an array containing all of the parents of the a previous category, then it will try to match the new parents to these previous parents.

    The program will output an array of parents for the input category, and create a folderStructure.xml file in the tmp/ directory that includes this hierarchy.
    It is able to append new hierarchies to whatever ones are currently in folderStructure.xml


IV] To exract the tar files (flat):
    ./extractImages.pl "dest_directory" "category"

    This program will scan current_directory/../img/category for tar files and transfer them to "dest_directory".
    The folders will be extracted into the same folder structure as the tar files have.
    If you wish to extract the images into a folder structure which represents the Word-Net hierarchy, run moveImages.pl

    There exists an extractAnnotations.pl file which should do the same thing, although it has not been updated recently.


V] To extract the tar files (structured):
    ./moveImages.pl "root_directory" "category" ("number of sub-categories")

    Depends on:
        findChildren.pl
        countImages.pl
        getPath.pl
        extractImages.pl
    
    This program will move images from current_directory/../img/category to root_directory/path where path represents the Word-Net hierarchy.
    You may pass "rt" as category (see part III)
    The final parameter is optional, and represents the number of sub-categories of the given category that you wish to move.
    If a number of sub-categories is given, the most populated categories will be returned. (See part III)

    As of now, the only tested parameter for this script is without specifying sub-categories and category != "rt".
    It should be possible to change the extractImages function call to extractAnnotations in order to put annotations (bounding boxes) within the given hierarchy.
   

VI] To get the structured directory of your extracted images:
    ./getChildPaths.pl "root_directory" "category"
    
    Depends on:
        findChildren.pl
        getPath.pl

    This program will return the path to category, which should be somewhere within the given root_directory.
    The program requires that there is a folderStructure.xml file within root_directory which outlines the folder structure.


VII] To get the distance between two categories:
    ./getDistances.pl "root_directory" "category1" "category2"

    This program will return the distance between category1 and category2. The math behind this distance is specified within the program's comments.
    The program assumes that there exists a folderStructure.xml file (created by IV or V) which outlines the hierarchy for these categories.

VIII] Other scripts:
    getPath.pl should return the path to a given category.
    extractAnnotations.pl should extract the bounding box tar file for a given category to a given directory.
    findChildren.pl returns all of the children of a given category, according to the Word-Net hierarchy.


END README.TXT
