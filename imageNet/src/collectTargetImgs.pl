#!/usr/bin/env perl

if ($ARGV[0] && $ARGV[1] && $ARGV[2] && $ARGV[3]) {
    $output = &collectTargetImgs($ARGV[0], $ARGV[1], $ARGV[2], $ARGV[3]);
} else {
    &ctiPostUsage();
}

sub ctiPostUsage () {
    die "\n\nUsage: ./collectTargetImgs.pl WNID in_path out_path num_images\n\tWNID can be a .txt file, a dir (of tar files), or a single ID\n\n\n";
}

sub collectTargetImgs ($$$$) {
    use warnings;
    use List::Util 'shuffle';
    use List::MoreUtils 'any';
    use POSIX;

    require 'findFiles.pl';
    require 'listChildren.pl';
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
    
    print "\ncollectTargetImgs: Copying $numToCpy target images from $inPath to $outPath\n";

#Ask user if the program should get the child nodes
    print "\ncollectTargetImgs: Would you like to extract the children of the input? [y/n] ";
    my $userChoice = <STDIN>;
    chomp($userChoice);
    my $correctAnswer = 0;
    while ($correctAnswer == 0) {
        if (($userChoice =~ m/^y$/) || ($userChoice =~ m/^n$/)) {
            $correctAnswer = 1;
            last;
        }
        print "collectTargetImgs: Please respond with 'y' or 'n': ";
        $userChoice = <STDIN>;
        chomp($userChoice);
    }
    print "\n";
    
#If getting child nodes, push nodes to list of WNIDs
    if ($userChoice =~ /^y$/) { #Grab children
        my @childArray;
        foreach my $parentWNID (@inArray) {
            my $HYPONYM_URL="http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid=[wnid]&full=1";

            print "collectTargetImgs: Downloading list of child synsets for $parentWNID...\n";
            $HYPONYM_URL =~ s/\[wnid\]/$parentWNID/;
            system("curl \"$HYPONYM_URL\" -# --cookie $TMP_DIR/cookies > $TMP_DIR/child_synsets.txt");  
            $HYPONYM_URL =~ s/$parentWNID/\[wnid\]/;
            print "collectTargetImgs: Done.\n\n";

            open(SYNSETS,"<","$TMP_DIR/child_synsets.txt") or die "Could not open $TMP_DIR/child_synsets.txt.\nERROR: $!\n";
            my @synsets= <SYNSETS>;
            close(SYNSETS);

            foreach my $childWNID (@synsets) {
                chomp($childWNID);
                $childWNID =~ s/\R//g; #Remove new-line
                $childWNID =~ s/^\-//g; #Remove - before WNID
                push(@childArray,$childWNID);
            }
        }
        push(@inArray,@childArray);
        undef @childArray;
    }

    my $ext = 'JPEG';
    my @totFileList = findFiles($inPath,$ext);
    my @wnidFileList;
    foreach my $WNID (@inArray) {
        foreach my $totFile (@totFileList) {
            if ($totFile =~ /$WNID/) {
                push(@wnidFileList,$totFile);
            }
        }
    }

#Check num images found
    $numImagesFound = scalar(@wnidFileList);
    print "collectTargetImgs: Found $numImagesFound images!\n";
    if ($numImagesFound < $numToCpy) {
        print "\ncollectTargetImgs: WARNING: Number of files found is less than the number requested. Copying the number found.\n";
        $numToCpy = $numImagesFound;
    }

#Find the number of categories & make a list of them
    my @categories;
    foreach my $wnidFile (@wnidFileList) { #look at each file in the file list
        if ($wnidFile =~ m/\/(n\d+)\_/) { #look for the synset id in the file name
            unless (any {/$1/} @categories) { #add category to @categories if it is not in there already
                push(@categories,$1);
            }
        }
    }
    my $numCategories = scalar(@categories);
    if ($userChoice =~ /^y$/) {
        print "collectTargetImgs: Found $numCategories categories!\n";
    }
    
    my $numExtraImgs = 0;
    my $numImgsPerCat = floor($numToCpy/$numCategories);
    if ($userChoice =~ /^y$/) {
        print "collectTargetImgs: Transfering about $numImgsPerCat images from each category.\n";
    }
    if (($numImgsPerCat*$numCategories) < $numToCpy) {
        $numExtraImgs = $numToCpy-($numCategories*$numImgsPerCat);
        print "collectTargetImgs: WARNING: $numExtraImgs extra images will be pulled from random categories.\n",
            "\tThis should be less than the number of categories = $numCategories\n\n";
    }
    
#Make a list of lists with indices [category][file]
    my (@fileLoL);
    for my $i (0 .. $numCategories-1) {
        my @matches = grep { /$categories[$i]/ } @wnidFileList;
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
            print "collectTargetImgs: WARNING: There are not enough images in the category $categories[$catIdx].\n",
                "\tThe target group will be unevenly weighted!\n";
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
    
    print "\ncollectTargetImgs: Finished moving $totCount images!\n";
}

1;
