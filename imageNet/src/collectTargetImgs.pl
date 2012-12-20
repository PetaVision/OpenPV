#!/usr/bin/env perl

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

    require 'findFiles.pl';
    require 'getFileExts.pl';
    #require 'listChildren.pl'; Not using currently
    require 'makeTempDir.pl';
    require 'checkInputType.pl';

#Set up vars and error check
    my $inWNID   = $_[0];
    my $inPath   = $_[1];
    my $outPath  = $_[2];
    my $numToCpy = $_[3];

#Set up temp dir
    my $TMP_DIR = makeTempDir();

#Set up input path
    unless (-d $inPath) {
        die "\n\nERROR: Input path must be a directory.\n";
    }
    $inPath =~ s/\/$//g;

#Set up output path
    system("mkdir -p $outPath") unless (-d $outPath);
    unless (-d $outPath) {
        die "\n\nERROR: Couldn't find or make output director $outPath.\n";
    }
    $outPath =~ s/\/$//g;

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
        use Archive::Tar;
        $tar = Archive::Tar->new;
        if ($numToCpy == -1) {
            print "collectTargetImgs: Extracting all of the images from tar files in $inPath to $outPath\n";
        } else {
            print "collectTargetImgs: Extracting $numToCpy images from tar files in $inPath to $outPath\n";
        }
    } else {
        if ($numToCpy == -1) {
            print "collectTargetImgs: Copying all of the target images from $inPath to $outPath\n";
        } else {
            print "collectTargetImgs: Copying $numToCpy target images from $inPath to $outPath\n";
        }
    }

#Ask user if the program should get the child nodes
    print "\ncollectTargetImgs: Would you like to extract the children of the input? [y/n] ";
    my $childChoice = <STDIN>;
    chomp($childChoice);
    my $correctAnswer = 0;
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
    
#If getting child nodes, push nodes to list of WNIDs (Requires different output than listChildren.pl provides)
    if ($childChoice =~ /^y$/) { #Grab children
        my @childArray;
        foreach my $parentWNID (@inArray) {
            #Download Image-Net children list if it does not already exist in the temp folder
            unless (-e "$TMP_DIR/child_synsets.txt") {
                my $HYPONYM_URL="http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid=[wnid]&full=1";
                print "collectTargetImgs: Downloading list of child synsets for $parentWNID...\n";
                $HYPONYM_URL =~ s/\[wnid\]/$parentWNID/;
                if ($useProxy) {
                    system("curl -x \"$PROXY_URL\" \"$HYPONYM_URL\" -# --cookie $TMP_DIR/cookies > $TMP_DIR/child_synsets.txt");  
                } else {
                    system("curl \"$HYPONYM_URL\" -# --cookie $TMP_DIR/cookies > $TMP_DIR/child_synsets.txt");  
                }
                $HYPONYM_URL =~ s/$parentWNID/\[wnid\]/;
                print "collectTargetImgs: Done.\n\n";
            }

            open(SYNSETS,"<","$TMP_DIR/child_synsets.txt") or die "Could not open $TMP_DIR/child_synsets.txt.\nERROR: $!\n";
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
    foreach my $WNID (@inArray) {
        foreach my $totFile (@totFileList) {
            if ($totFile =~ /$WNID/) {
                push(@wnidFileList,$totFile);
            }
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
            die "\ncollectTargetImgs: ERROR: Number of images found is 0.\n";
        }
    }

#Find the number of categories & make a list of them
    my @categories;
    foreach my $wnidFile (@wnidFileList) { #look at each file in the file list
        if ($wnidFile =~ m/\/(n\d+)/) { #look for the synset id in the file name
            unless (grep {$1 eq $_} @categories) { #add category to @categories if it is not in there already
                push(@categories,$1);
            }
        }
    }
    my $numCategories = scalar(@categories);
    if ($numCategories < 1) {
        die "collectTargetImgs: ERROR: number of categories is less than 1!\n";
    }
    print "collectTargetImgs: Found $numCategories categories.\n";
    
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
        die "\n\nERROR: Number of categories does not match LoL size.\n\n";
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
    
        my $count = 0;
        my @fileAry = 0 .. $numCatFiles-1;

        if ($numExtraImgs > 0) { #if we need to get an uneven amount of images from the categories
            if ($numCatFiles >= $numImgsPerCat+1) { #if this category has enough images to grab one extra
                while ($count < $numImgsPerCat+1) { #grab one more image than usual
                    my $file = $fileLoL[$catAry[$catNum]][$fileAry[$count]];
                    if ($ext =~ /^JPEG$/) {
                        system("cp \"$file\" \"$outPath\"");
                    } else {
                        my $outFile = $outPath."/".$file;
                        $tar->read($wnidFileList[$catAry[$catNum]]);
                        $tar->extract_file($file,$outFile);
                    }
                    $count += 1;
                    $percComp = 100*(($totCount+$count)/$numToCpy);
                    print "collectTargetImgs: Percent Complete: $percComp %    \r";
                }
                $numExtraImgs -= 1;
            }
        } else {
            while ($count < $numImgsPerCat && $count < $numCatFiles) {
                my $file = $fileLoL[$catAry[$catNum]][$fileAry[$count]];
                if ($ext =~ /^JPEG$/) {
                    system("cp \"$file\" \"$outPath\"");
                } else {
                    my $outFile = $outPath."/".$file;
                    $tar->read($wnidFileList[$catAry[$catNum]]);
                    $tar->extract_file($file,$outFile);
                }
                $count += 1;
                $percComp = 100*(($totCount+$count)/$numToCpy);
                print "collectTargetImgs: Percent Complete: $percComp %    \r";
            }
        }
        push(@numImgsPerCat,$count);
        $totCount += $count;
    }
    print "collectTargetImgs: Percent Complete: 100 %\n";
    print "-------\ncollectTargetImgs: Images per category:\n";
    for my $catNum (0 .. $numCategories-1) {
        print "\t$categories[$catAry[$catNum]] :: $numImgsPerCat[$catNum]\n";
    }
    print "-------\ncollectTargetImgs: Moved $totCount out of $numToCpy images!\n";
}


1;
