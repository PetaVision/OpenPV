#!/usr/bin/env perl

############
## collectDistractorImgs.pl
##
## Written by:
##      Dylan Paiton
##      Los Alamos National Laboratory, Group ISR-2 
##      paiton@lanl.gov
##
## Collect random distractor images from input folder
##
## TODO:
##
##
############

if ($ARGV[0] && $ARGV[1] && $ARGV[2]) {
    if ($ARGV[3]) {
        $output = &collectDistractorImgs($ARGV[0], $ARGV[1], $ARGV[2], $ARGV[3]);
    } else {
        $output = &collectDistractorImgs($ARGV[0], $ARGV[1], $ARGV[2],"NULL");
    }
} else {
    &cdiPostUsage();
}

sub cdiPostUsage () {
    die "\n\nUsage: ./collectDistractorImgs.pl in_path out_path num_images [exclude_WNID]\n\texclude_WNID can be an ID, text file, or folder\n\n\n";
};

sub collectDistractorImgs ($$$$) {
    #use warnings; ##Currently warns about two unused vars. This is not a problem.

    use globalVars;
    my $useProxy = getUseProxy globalVars();
    my $PROXY_URL = "";
    if ($useProxy) {
        $PROXY_URL = getProxyURL globalVars();
    }

    use List::Util 'shuffle';
    use List::MoreUtils 'any';
    use POSIX;

    require 'findFiles.pl';
    require 'listChildren.pl';
    require 'listParents.pl';
    require 'makeTempDir.pl';
    require 'checkInputType.pl';

#Set up temp dir
    my $TMP_DIR = makeTempDir();

#Set up input vars and error check
    my $inPath   = $_[0];
    my $outPath  = $_[1];
    my $numToCpy = $_[2];

    my $inputType = 0;
    if ($_[3]) {
        $excludeWNID = $_[3];
        chomp($excludeWNID);

        $inputType = checkInputType($excludeWNID);
        if ($inputType == 0) {
            &cdiPostUsage();
        }
    } else {
        $excludeWNID = "NULL";
    }

    my @excludeArray;
    if ($inputType==1) { #The user has input a synset id
        @excludeArray = ($excludeWNID);
    } elsif ($inputType==2) { #The user has input a text file full of synset ids
        #Open input file
        open(INTEXT,"<", $excludeWNID) or die $!;
        @excludeArray = <INTEXT>;
        close(INTEXT);
    } elsif ($inputType==3) { #The user has input a dir full of tar files
        $excludeWNID =~ s/\/$//g;
        $excludeWNID =~ s/\s/\\ /g;

        print "collectDistractorImgs: Getting a list of WNIDS from the tar files in the input dir $excludeWNID\n";
        my $ext = 'tar';
        my @fileList = findFiles($excludeWNID,$ext);

        for (my $i=0; $i<scalar(@fileList); $i++) {
            if ($fileList[$i] =~ /(n\d+)/) {
                $fileList[$i] = $1;
            } else {
                delete $fileList[$i];
            }
        }

        my $numFilesFound = scalar(@fileList);
        print "collectDistractorImgs: Found $numFilesFound WNIDs to exclude.\n";

        @excludeArray = @fileList;
    }

    unless (-d $inPath) {
        die "\n\nERROR: Input path must be a directory.\n";
    }

    system("mkdir -p $outPath") unless (-d $outPath);
    unless (-d $outPath) {
        die "\n\nERROR: Couldn't find or make output director $outPath.\n";
    }

    print "\ncollectDistractorImgs: Copying $numToCpy random images from $inPath to $outPath\n";
    $inPath =~ s/\/$//g;
    $outPath =~ s/\/$//g;

    my $ext = 'JPEG';
    my @fileList = findFiles($inPath,$ext);

    $numImagesFound = scalar(@fileList);
    print "collectDistractorImgs: Found $numImagesFound images!\n";
    if ($numImagesFound < $numToCpy) {
        print "\ncollectDistractorImgs: WARNING: Number of files found is less than the number requested. Copying the number found.\n";
        $numToCpy = $numImagesFound;
    }

    #If exclude WNID is listed, make a list of it, all of its parents and all of its children
    my @excludeSet;
    unless ($excludeWNID =~ "NULL") { #If user has given an exclude ID
        foreach $WNID (@excludeArray) {
            print "collectDistractorImgs: Excluding $WNID and its child/parent categories from the distractor set.\n";
            ($chNamesRef, $chIdsRef) = listChildren($WNID);
            ($paNamesRef, $paIdsRef) = listParents($WNID);
            push(@excludeSet,@$chIdsRef);
            push(@excludeSet,@$paIdsRef);
        }
    }

    #Find the number of categories & make a list of them
    my @categories;
    foreach my $file (@fileList) { #look at each file in the file list
        if ($file =~ m/\/(n\d+)\_/) { #look for the synset id in the file name
            unless (any {/$1/} @categories) { #add category to @categories if it is not in there already
                unless (any {/$1/} @excludeSet) {#add category to @categories if it is not in the exclude set
                    push(@categories,$1);
                }
            }
        }
    }
    my $numCategories = scalar(@categories);
    print "collectDistractorImgs: Found $numCategories categories!\n";
    
    my $numExtraImgs = 0;
    my $numImgsPerCat = floor($numToCpy/$numCategories);
    print "collectDistractorImgs: Transfering about $numImgsPerCat images from each category.\n";
    if (($numImgsPerCat*$numCategories) < $numToCpy) {
        $numExtraImgs = $numToCpy-($numCategories*$numImgsPerCat);
        print "collectDistractorImgs: WARNING: $numExtraImgs extra images will be pulled from random categories.\n",
            "\tThis should be less than the number of categories = $numCategories\n\n";
    }
    
    #Make a list of lists with indices [category][file]
    my (@fileLoL);
    for my $i (0 .. $numCategories-1) {
        my @matches = grep { /$categories[$i]/ } @fileList;
        push(@fileLoL,[@matches]);
    }
    
    #Sanity check
    unless ($#fileLoL+1 == $numCategories) {
        die "\n\nERROR: Number of categories does not match LoL size.\n\n";
    }
    
    #Shuffle categories
    my @shuffledLoL = shuffle(@fileLoL);
    
    ####################################################################
    ## Uncomment below to see the total number of files per category
    ####################################################################
    #print "\n------------------\nNumber of files per category:\n";
    #my $idx = 0;
    #for $subArryRef (@shuffledLoL) {
    #    my @subFileList = @$subArryRef;
    #    my $numCatFiles = scalar(@subFileList);
    #    print "\t$idx: $numCatFiles\n";
    #    $idx += 1;
    #}
    #print "------------------\n\n";
    ####################################################################
    
    #Shuffle images in each category and copy
    my $catIdx   = 0;
    my $totCount = 0;
    for $subArryRef (@shuffledLoL) {
        my @subFileList = @$subArryRef;
        my $numCatFiles = scalar(@subFileList);
        unless ($numCatFiles >= $numImgsPerCat) {
            print "collectDistractorImgs: WARNING: There are not enough images in the category $categories[$catIdx].\n",
                "\tThe distractor group will be unevenly weighted!\n";
        }
    
        my $count = 0;
        my @shuffledImgList = shuffle(@subFileList);
    
        if ($numExtraImgs > 0) { #if we need to get an uneven amount of images from the categories
            if ($numCatFiles >= $numImgsPerCat+1) { #if this category has enough images to grab one extra
                while ($count < $numImgsPerCat+1) { #grab one more image than usual
                    my $file = $shuffledImgList[$count];
                    system("cp \"$file\" \"$outPath\"");
                    $count += 1;
                }
                $numExtraImgs -= 1;
            }
        } else {
            while ($count < $numImgsPerCat && $count < $numCatFiles) {
                my $file = $shuffledImgList[$count];
                system("cp \"$file\" \"$outPath\"");
                $count += 1;
            }
        }
    
        $totCount += $count;
        $catIdx += 1;
    }
    
    print "\ncollectDistractorImgs: Finished moving $totCount images!\n";
}

1;
