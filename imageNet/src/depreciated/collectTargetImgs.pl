#!/usr/bin/env perl

############
## collectTargetImgs.pl
##
## Written by:
##      Dylan Paiton
##      Los Alamos National Laboratory, Group ISR-2 
##      paiton@lanl.gov
##
## Collect random target images from input folder
##
## TODO:
##
##
############

if ($ARGV[0] && $ARGV[1] && $ARGV[2] && $ARGV[3]) {
    $output = &collectTargetImgs($ARGV[0], $ARGV[1], $ARGV[2], $ARGV[3]);
} else {
    &ctiPostUsage();
}

sub ctiPostUsage () {
    die "\n\nUsage: ./collectTargetImgs.pl WNID in_path out_path num_images\n\tWNID can be a .txt file, a dir (of tar files), or a single ID\n\tnum_images can be '-1' for all images\n\n\n";
}

sub collectTargetImgs ($$$$) {
    use warnings;
    use strict;

    use globalVars;
    my $useProxy = getUseProxy globalVars();
    my $PROXY_URL = "";
    if ($useProxy) {
        $PROXY_URL = getProxyURL globalVars();
    }

    use List::Util 'shuffle';
    use POSIX;
    use Archive::Tar;

    require 'findFiles.pl';
    require 'findFolders.pl';
    require 'getFileExts.pl';
    #require 'listChildren.pl'; #This code is not implemented such that it works with listChildren.pl
    require 'makeTempDir.pl';
    require 'checkInputType.pl';

#Set up vars and error check
    my $inWNID   = $_[0];
    my $inPath   = $_[1];
    my $outPath  = $_[2];
    my $numToCpy = $_[3];

    $outPath =~ s/\/$//g;

#Set up temp dir
    my $TMP_DIR = makeTempDir();

#Set up input path
    unless (-d $inPath) {
        die "\n\nERROR: Input path must be a directory.\n";
    }
    $inPath =~ s/\/$//g;

#Set up output path
    my $outImgPath = $outPath."/Images";
    system("mkdir -p $outImgPath") unless (-d $outImgPath);
    unless (-d $outImgPath) {
        die "\n\nERROR: Couldn't find or make output director $outImgPath.\n";
    }
    $outImgPath =~ s/\/$//g;

#Check to see if the user has input a WNID, a folder, or a text file listing WNIDs
    my $inputType = checkInputType($inWNID);

#Read input WNID(s)
    my @inArray;
    if ($inputType==1) { #The user has input a synset id
        @inArray = ($inWNID);
    } elsif ($inputType==2) { #The user has input a text file full of synset ids
        #Open input file
        open(INTEXT,"<", $inWNID) or die $!;
        @inArray = <INTEXT>;
        close(INTEXT);
    } elsif ($inputType==3) { #The user has input a dir full of tar files
        $inWNID =~ s/\/$//g;
        $inWNID =~ s/\s/\\ /g;

        print "collectTargetImgs: Getting a list of WNIDS from the tar files in the input dir $inWNID\n";
        my $ext = 'tar';
        my @fileList = findFiles($inWNID,$ext);

        for (my $i=0; $i<scalar(@fileList); $i++) {
            if ($fileList[$i] =~ /(n\d+)/) {
                $fileList[$i] = $1;
            } else {
                delete $fileList[$i];
            }
        }

        my $numFilesFound = scalar(@fileList);
        print "collectTargetImgs: Found $numFilesFound WNIDs.\n";

        @inArray = @fileList;
    } else {
        &ctiPostUsage();
    }
    
#Find out if input path contains tar files or jpeg files
    my @fileExts = getFileExts($inPath);
    my $numFileExts = scalar(@fileExts);
    unless ($numFileExts == 1) {
        die "collectTargetImgs: ERROR: Found more than one file extension in $inPath.\n";
    }
    my $ext = $fileExts[0];

    my $tar;
    if ($ext =~ /^tar$/) { #Set up for extracting tar files
        $tar = Archive::Tar->new;
        if ($numToCpy == -1) {
            print "collectTargetImgs: Extracting all of the images from tar files in $inPath to $outImgPath\n";
        } else {
            print "collectTargetImgs: Extracting $numToCpy images from tar files in $inPath to $outImgPath\n";
        }
    } else {
        if ($numToCpy == -1) {
            print "collectTargetImgs: Copying all of the target images from $inPath to $outImgPath\n";
        } else {
            print "collectTargetImgs: Copying $numToCpy target images from $inPath to $outImgPath\n";
        }
    }

#Ask user if the program should only transfer images with(out) bounding boxes
##TODO: have default path releative to input image path
    print "\ncollectTargetImgs: Would you like to extract [1] only images with BBs [2] only images without BBs or [3] images with and without BBs? [1/2/3] ";
    my $bbChoice = <STDIN>;
    chomp($bbChoice);
    my $correctAnswer = 0;
    while ($correctAnswer == 0) {
        if (($bbChoice =~ m/^1$/) || ($bbChoice =~ m/^2$/) || ($bbChoice =~ m/^3$/)) {
            $correctAnswer = 1;
            last;
        }
        print "collectTargetImgs: Please respond with '1'(BBs), '2'(no BBs), or '3'(mixed): ";
        $bbChoice = <STDIN>;
        chomp($bbChoice);
    }
    print "\n";

#Ask user if the program should get the child nodes
    print "\ncollectTargetImgs: Would you like to extract the children of the input? [y/n] ";
    my $childChoice = <STDIN>;
    chomp($childChoice);
    $correctAnswer = 0;
    while ($correctAnswer == 0) {
        if (($childChoice =~ m/^y$/) || ($childChoice =~ m/^n$/)) {
            $correctAnswer = 1;
            last;
        }
        print "collectTargetImgs: Please respond with 'y' or 'n': ";
        $childChoice = <STDIN>;
        chomp($childChoice);
    }
    print "\n";
    
#Get BB list if user specified a BB preference
    my $bbExt;
    my $bbPath;
    my $outBBPath;
    if (($bbChoice =~ m/^1$/) || ($bbChoice =~ m/^2$/)) {
        $outBBPath = $outPath."/Annotations";
        system("mkdir -p $outBBPath") unless (-d $outBBPath);
        unless (-d $outBBPath) {
            die "\n\nERROR: Couldn't find or make output directory $outBBPath.\n";
        }
        $outBBPath =~ s/\/$//g;

        print "collectTargetImgs: Please enter the path to the Bounding Boxes: ";
        $bbPath = <STDIN>;
        chomp($bbPath);
        print "\n";

        unless (-d $bbPath) {
            die "collectTargetImgs: ERROR: Couldn't find bounding box directory $bbPath.\n";
        }

        print "collectTargetImgs: Searching $bbPath for annotation file extensions.\n";
        my @bbFileExts = getFileExts($bbPath);
        my $bbNumFileExts = scalar(@bbFileExts);
        unless ($bbNumFileExts == 1) {
            die "collectTargetImgs: ERROR: Found more than one ($bbNumFileExts) file extension in $bbPath.\nExtensions found: \n".join("\n\t",@bbFileExts)."\n";
            #die "collectTargetImgs: ERROR: Found more than one ($bbNumFileExts) file extension in $bbPath.";
        }
        $bbExt = $bbFileExts[0];
    }

#If getting child nodes, push nodes to list of WNIDs (Requires different output than listChildren.pl provides)
    if ($childChoice =~ /^y$/) { #Grab children
        my @childArray;
        foreach my $parentWNID (@inArray) {
            #Download Image-Net children list if it does not already exist in the temp folder
            my $synsetOutFile = $TMP_DIR."/".$parentWNID."_child_synsets.txt";
            unless (-e $synsetOutFile) {
                my $HYPONYM_URL="http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid=[wnid]&full=1";
                print "collectTargetImgs: Downloading list of child synsets for $parentWNID...\n";
                $HYPONYM_URL =~ s/\[wnid\]/$parentWNID/;
                if ($useProxy) {
                    system("curl -x \"$PROXY_URL\" \"$HYPONYM_URL\" -# --cookie $TMP_DIR/cookies > $synsetOutFile");  
                } else {
                    system("curl \"$HYPONYM_URL\" -# --cookie $TMP_DIR/cookies > $synsetOutFile");  
                }
                $HYPONYM_URL =~ s/$parentWNID/\[wnid\]/;
                print "collectTargetImgs: Done.\n\n";
            }

            open(SYNSETS,"<",$synsetOutFile) or die "Could not open $synsetOutFile.\nERROR: $!\n";
            my @synsets= <SYNSETS>;
            close(SYNSETS);

            foreach my $childWNID (@synsets) {
                chomp($childWNID);
                $childWNID =~ s/\R//g; #Remove new-line
                $childWNID =~ s/^\-//g; #Remove - before WNID
                unless ($childWNID eq $parentWNID) {
                    push(@childArray,$childWNID);
                }
            }
        }
        push(@inArray,@childArray);
        undef @childArray;
    }

#Get list of files for the desired WNIDs.
    my @totFileList = findFiles($inPath,$ext);
    my @wnidFileList;
    my @categories;
    foreach my $WNID (@inArray) {
        foreach my $totFile (@totFileList) {
            if ($totFile =~ /$WNID/) {
                push(@wnidFileList,$totFile);
                if ($totFile=~ m/\/(n\d+)/) { #look for the synset id in the file name
                    unless (grep {$1 eq $_} @categories) { #add category to @categories if it is not in there already
                        push(@categories,$1);
                    }
                }
            }
        }
    }

#Get list of files for the desired BBs.
    my @BBFileList;
    if (($bbChoice =~ m/^1$/) || ($bbChoice =~ m/^2$/)) { 
        print "collectTargetImgs: Correlating images with respective bounding boxes.\n";
        my @BBCats;
        my @totBBFileList;
        if ($bbExt =~ m/xml/) {
            @totBBFileList = findFolders($bbPath,$bbExt);
        } else {
            @totBBFileList = findFiles($bbPath,$bbExt);
        }

        if (scalar(@totBBFileList) == 0) {
            die "collectTargetImgs: ERROR: Did not find any bounding boxes for input path $bbPath and extension $bbExt.\n";
        }
        foreach my $WNID (@categories) {
            foreach my $totBBFile (@totBBFileList) {
                if ($totBBFile =~ /$WNID/) {
                    push(@BBCats,$WNID);
                    push(@BBFileList,$totBBFile);
                }
            }
        }

        if ($bbChoice =~ m/^1$/) {
            ##List of cats that have BBs with them
            my @newCats;
            foreach my $BBCat (@BBCats) {
                foreach my $realCat (@categories) {
                    if ($BBCat =~ /^$realCat$/) {
                        push(@newCats,$BBCat);
                    }
                }
            }

            ##List of WNIDS with BBs with them
            my @newWNIDFileList;
            foreach my $BBFile (@BBFileList) {
                foreach my $WNIDFile (@wnidFileList) {
                    $BBFile =~ m/(n\d+)/;
                    if ($WNIDFile =~ /$1/) {
                        push(@newWNIDFileList,$WNIDFile);
                    }
                }
            }

            ##Need to modify @categories and $wnidFileList to only include categorys and files with BBs
            @categories = @newCats;
            @wnidFileList = @newWNIDFileList;

            undef @newCats;
            undef @newWNIDFileList;
        }
    }

#Check num files found
    my $numImagesFound;
    if ($ext =~ /^JPEG$/) {
        $numImagesFound = scalar(@wnidFileList);
    } else {
        $numImagesFound = 0;
        foreach my $tarFile (@wnidFileList) {
            $tar->read($tarFile);
            my @fileList = $tar->list_files();
            $numImagesFound += scalar(@fileList);
        }
    }
    if ($numToCpy == -1) {
        $numToCpy = $numImagesFound;
    }
    print "collectTargetImgs: Found $numImagesFound images.\n";
    if ($numImagesFound < $numToCpy) {
        if ($numImagesFound > 0) {
            print "\ncollectTargetImgs: WARNING: Number of images found is less than the number requested. Copying the number found.\n";
            $numToCpy = $numImagesFound;
        } else {
            die "\ncollectTargetImgs: ERROR: Number of images found is $numImagesFound.\n";
        }
    }

#Check number of categories
    my $numCategories = scalar(@categories);
    if ($numCategories < 1) {
        die "collectTargetImgs: ERROR: number of categories is less than 1!\n";
    }
    print "collectTargetImgs: Found $numCategories categories.\n";
    
#Compute number of extra images needed
    my $numExtraImgs = 0;
    my $numImgsPerCat = floor($numToCpy/$numCategories);
    if ($childChoice =~ /^y$/) {
        print "collectTargetImgs: Transfering about $numImgsPerCat images from each category.\n";
    }
    if (($numImgsPerCat*$numCategories) < $numToCpy) {
        $numExtraImgs = $numToCpy-($numCategories*$numImgsPerCat);
        print "collectTargetImgs: WARNING: $numExtraImgs extra images will be pulled from random child categories.\n",
            "\tThis should be less than the number of categories = $numCategories\n\n";
    }

#Make a list of lists with indices [category][file]
    my @fileLoL;
    for my $i (0 .. $numCategories-1) {
        my @matches;
        if ($ext =~ /^JPEG$/) {
            @matches = grep { /$categories[$i]/ } @wnidFileList;
        } else {
            $tar->read($wnidFileList[$i]);
            @matches = $tar->list_files();
        }
        push(@fileLoL,[@matches]);

    }

#Sanity check
    unless ($#fileLoL+1 == $numCategories) {
        die "\n\ncollectTargetImgs: ERROR: Number of categories does not match LoL size.\n\n";
    }

#Shuffle images categories and copy
    my @numImgsPerCat;
    my @catAry   = 0 .. $numCategories-1;
    my $totCount = 0;
    print "collectTargetImgs: Moving images...\n";
    for my $catNum (0 .. $numCategories-1) {
        my $percComp = 100*($totCount/$numToCpy);
        print "collectTargetImgs: Percent Complete: $percComp %    \r";

        my $subFileListRef = @fileLoL[$catAry[$catNum]];
        my @subFileList = @$subFileListRef;

        my $numCatFiles = scalar(@subFileList);
        unless ($numCatFiles >= $numImgsPerCat) {
            print "collectTargetImgs: WARNING: There are not enough images in the category $categories[$catAry[$catNum]].\n",
                "\tThe target group will be unevenly weighted!\n";
        }
    
        my $successCount = 0;
        my $attemptCount = 0;
        my @fileAry      = 0 .. $numCatFiles-1;

        if ($numExtraImgs > 0) { #if we need to get an uneven amount of images from the categories
            if ($numCatFiles >= $numImgsPerCat+1) { #if this category has enough images to grab one extra
                while ($successCount+$attemptCount < $numImgsPerCat+1) { #grab one more image than usual
                    my $file = $fileLoL[$catAry[$catNum]][$fileAry[$successCount+$attemptCount]];

                    my $bbSuccess = 1;
                    if (($bbChoice =~ m/^1$/) || ($bbChoice =~ m/^2$/)) {
                        my $bbWNID = $file;
                        my $bbFileName = $file;

                        $bbWNID =~ s/([\w\d]+)_[\.\w]+/$1/; #Pulls out just the WNID, before the underscore
                        $bbFileName =~ s/([\w\d]+_[\d]+)[\.\w]+/$1/; #Pulls out the full file-name, including the underscore

                        my $bbFile;
                        my $outFile = $outBBPath."/".$bbFileName.".xml";

                        ##Get Bounding Box
                        if ($catAry[$catNum] < scalar(@BBFileList)) {
                            $bbFile = "Annotation/".$bbWNID."/".$bbFileName.".xml";
                            if ($bbExt =~ m/\.tar\.gz/) {
                                $tar->read($BBFileList[$catAry[$catNum]],'tgz');
                                $bbSuccess = $tar->contains_file($bbFile);
                                if ($bbChoice =~ m/^1$/) {
                                    if ($bbSuccess) {
                                        $tar->extract_file($bbFile,$outFile);
                                    }
                                } 
                            } elsif ($bbExt =~ m/^xml$/) {
                                $bbFile = $bbPath."/".$bbWNID."/".$bbFileName.".xml";
                                $bbSuccess = -e $bbFile;
                                if ($bbChoice =~ m/^1$/) {
                                    if ($bbSuccess) {
                                        system("cp \"$bbFile\" \"$outFile\"");
                                    }
                                }
                            }
                        } else {
                            $bbSuccess = 0;
                        }
                    }

                    ##If we got the box and we wanted the box, or we didn't get the box and we didn't want the box
                    if ((($bbSuccess) && ($bbChoice =~ m/^1$/)) || ((!$bbSuccess) && ($bbChoice =~ m/^2$/)) || ($bbChoice =~ m/^3$/)) {
                        ##Get actual image
                        if ($ext =~ /^JPEG$/) {
                            system("cp \"$file\" \"$outImgPath\"");
                        } else {
                            my $outFile = $outImgPath."/".$file;
                            $tar->read($wnidFileList[$catAry[$catNum]]);
                            $tar->extract_file($file,$outFile);
                        }

                        $successCount += 1;
                        $percComp = 100*(($totCount+$successCount)/$numToCpy);
                        print "collectTargetImgs: Percent Complete: $percComp %                        \r";
                    } else {
                        $attemptCount += 1;
                    }
                }
                $numExtraImgs -= 1;
            }
        } else {
            while (($successCount < $numImgsPerCat) && ($successCount+$attemptCount < $numCatFiles)) {
                my $file = $fileLoL[$catAry[$catNum]][$fileAry[$successCount+$attemptCount]];

                my $bbSuccess = 1;
                if (($bbChoice =~ m/^1$/) || ($bbChoice =~ m/^2$/)) {
                    my $bbWNID = $file;
                    my $bbFileName = $file;

                    $bbWNID =~ s/([\w\d]+)_[\.\w]+/$1/; #Pulls out just the WNID, before the underscore
                    $bbFileName =~ s/([\w\d]+_[\d]+)[\.\w]+/$1/; #Pulls out the full file-name, including the underscore

                    my $bbFile;
                    my $outFile = $outBBPath."/".$bbFileName.".xml";

                    ##Get Bounding Box
                    if ($catAry[$catNum] < scalar(@BBFileList)) {
                        $bbFile = "Annotation/".$bbWNID."/".$bbFileName.".xml";
                        if ($bbExt =~ m/\.tar\.gz/) {
                            $tar->read($BBFileList[$catAry[$catNum]],'tgz');
                            $bbSuccess = $tar->contains_file($bbFile);
                            if ($bbChoice =~ m/^1$/) {
                                if ($bbSuccess) {
                                    $tar->extract_file($bbFile,$outFile);
                                }
                            } 
                        } elsif ($bbExt =~ m/^xml$/) {
                            $bbFile = $bbPath."/".$bbWNID."/".$bbFileName.".xml";
                            $bbSuccess = -e $bbFile;
                            if ($bbChoice =~ m/^1$/) {
                                if ($bbSuccess) {
                                    system("cp \"$bbFile\" \"$outFile\"");
                                }
                            }
                        }
                    } else {
                        $bbSuccess = 0;
                    }
                }

                ##If we got the box and we wanted the box, or we didn't get the box and we didn't want the box
                if ((($bbSuccess) && ($bbChoice =~ m/^1$/)) || ((!$bbSuccess) && ($bbChoice =~ m/^2$/)) || ($bbChoice =~ m/^3$/)) {
                    ##Get actual image
                    if ($ext =~ /^JPEG$/) {
                        system("cp \"$file\" \"$outImgPath\"");
                    } else {
                        my $outFile = $outImgPath."/".$file;
                        $tar->read($wnidFileList[$catAry[$catNum]]);
                        $tar->extract_file($file,$outFile);
                    }

                    $successCount += 1;
                    $percComp = 100*(($totCount+$successCount)/$numToCpy);
                    print "collectTargetImgs: Percent Complete: $percComp %                        \r";
                } else {
                    $attemptCount += 1;
                }
            }
        }
        push(@numImgsPerCat,$successCount);
        $totCount += $successCount;
    }
    print "collectTargetImgs: Percent Complete: 100 %\n";
    print "-------\ncollectTargetImgs: Images per category:\n";
    for my $catNum (0 .. $numCategories-1) {
        print "\t$categories[$catAry[$catNum]] :: $numImgsPerCat[$catNum]\n";
    }
    print "-------\ncollectTargetImgs: Moved $totCount out of $numToCpy images!\n";
}


1;
