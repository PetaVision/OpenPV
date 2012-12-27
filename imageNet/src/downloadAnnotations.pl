#!/usr/bin/env perl

############
## downloadAnnotations.pl
##
## Written by:
##      Dylan Paiton
##      Los Alamos National Laboratory, Group ISR-3 
##      paiton@lanl.gov
## 
## Extracts $inputAnnotations from $ARCH_DIR to $destDir
##  Inputs:
##      $inputCategory = desired category synset ID (can be ID, text file, or folder)
##      $inputDir      = directory containing annotation archive files
##      $destDir       = desired output directory
##          *Note: There is no need to escape spaces for $destDir.
############

#####
##Uncomment below to run from command line
##This must stay commented in order to call this function from another program
if ($ARGV[0] && $ARGV[1]) {
    $out = &downloadAnnotations($ARGV[0], $ARGV[1]);
    print "downloadAnnotations: Program complete.\n";
} else {
    &daPostUsage();
}
#####

sub daPostUsage () {
    die "\n\nUsage: ./downloadAnnotations.pl WNID destination_dir\n\tWNID can be a text file, a folder path, or a single ID\n\n\n";
}

sub downloadAnnotations ($$) {
    use warnings;

    use globalVars;
    my $useProxy  = getUseProxy globalVars();
    my $PROXY_URL = "";
    if ($useProxy) {
        $PROXY_URL = getProxyURL globalVars();
    }

    require 'listChildren.pl';
    require 'findFiles.pl';
    require 'makeTempDir.pl';
    require 'checkInputType.pl';

    my $USER_AGENT = "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)";

    my $inputCategory = $_[0];
    chomp($inputCategory);
    my $destDir   = $_[1];
    chomp($destDir);

#Set up temp dir
    my $TMP_DIR = makeTempDir();

#Clean up category
    $inputCategory =~ s/\s/\\ /g;

#Clean up dest dir
    $destDir =~ s/\/$//g;
    $destDir =~ s/\s/\\ /g;
    system("mkdir -p $destDir") unless (-d $destDir);

#Ask user if the program should get the child nodes
    my $getChildren = 'n';
    print "\ndownloadAnnotations: Would you like to download the child synset annotations? [y/n] ";
    my $userChoice = <STDIN>;
    chomp($userChoice);
    my $correctAnswer = 0;
    while ($correctAnswer == 0) {
        if (($userChoice =~ m/^y$/) || ($userChoice =~ m/^n$/)) {
            $correctAnswer = 1;
            last;
        }
        print "downloadAnnotations: Please respond with 'y' or 'n': ";
        $userChoice = <STDIN>;
        chomp($userChoice);
    }
    if ($userChoice =~ /y/) {
        print "downloadAnnotations: Downloading child synset annotations.\n";
        $getChildren = 'y';
    } else {
        print "downloadAnnotations: Not downloading child synset annotations.\n";
        $getChildren = 'n';
    }

#Check to see if the user has input a WNID, a folder, or a text file listing WNIDs
    my $inputType = checkInputType($inputCategory);
    my @inArray;
    if ($inputType==1) { #The user has input a synset id
        @inArray = ($inputCategory);
    } elsif ($inputType==2) { #The user has input a text file full of synset ids
        #Open input file
        open(INTEXT,"<", $inputCategory) or die $!;
        @inArray = <INTEXT>;
        close(INTEXT);
    } elsif ($inputType==3) { #The user has input a dir full of tar files
        $inputCategory =~ s/\/$//g;
        $inputCategory =~ s/\s/\\ /g;
        print "downloadAnnotations: Getting list of files from the input dir $inputCategory.\n";
        my $ext = 'tar';
        my @fileList = findFiles($inputCategory,$ext);

        for (my $i=0; $i<scalar(@fileList); $i++) {
            if ($fileList[$i] =~ /(n\d+)/) {
                $fileList[$i] = $1;
            } else {
                delete $fileList[$i];
            }
        }

        my $numFilesFound = scalar(@fileList);
        print "downloadAnnotations: Found $numFilesFound WNIDs to find annotations for.\n";

        @inArray = @fileList;
    } else { #shouldn't happen
        &eiPostUsage();
    }

    #Get children of each WNID if needed
    if ($getChildren =~ /y/) {
        print "downloadAnnotations: Adding children to download list.\n";
        my @kidsArray;
        foreach my $WNID (@inArray) {
            my ($namesRef, $idsRef) = &listChildren($ARGV[0]);
            my @names = @$namesRef;
            my @WNIDs = @$idsRef;
            push(@kidsArray,@WNIDs);
        }
        push (@inArray,@kidsArray);
    }

    my %seen = ();
#Parse through input items, everything inside this loop is done for each input item
    foreach my $WNID (@inArray) {
        next if $seen{$WNID}++; #Array probably has duplicates
        
        print "\n-------------------\n";
        chomp($WNID);
        my $BBOX_URL1  = "http://www.image-net.org/api/download/imagenet.bbox.synset?wnid=";
        my $BBOX_URL2  ="http://www.image-net.org/downloads/bbox/[wnid].tar.gz";

        #Check to make sure bounding boxes exist
        if ($use_proxy) {
            system("curl -x \"$PROXY_URL\" \"$BBOX_URL1$WNID\" -s -L --cookie $TMP_DIR/cookies --cookie-jar $TMP_DIR/cookies -A \"$USER_AGENT\" 1> $TMP_DIR/BBout.log");
        } else {
            system("curl \"$BBOX_URL1$WNID\" -s -L --cookie $TMP_DIR/cookies --cookie-jar $TMP_DIR/cookies -A \"$USER_AGENT\" 1> $TMP_DIR/BBout.log");
        }
        $BBOX_URL2 =~ s/\[wnid\]/$WNID/;
        if ($use_proxy) {
            system("curl -x \"$PROXY_URL\" \"$BBOX_URL2\" -s --cookie $TMP_DIR/cookies --cookie-jar $TMP_DIR/cookies -e \"$BBOX_URL1$WNID\" -I 1> $TMP_DIR/BBheaddump.log");
        } else {
            system("curl \"$BBOX_URL2\" -s --cookie $TMP_DIR/cookies --cookie-jar $TMP_DIR/cookies -e \"$BBOX_URL1$WNID\" -I 1> $TMP_DIR/BBheaddump.log");
        }
        $BBOX_URL2 =~ s/$WNID/\[wnid\]/;

        open(BBHEADDUMP,"<","$TMP_DIR/BBheaddump.log") or die $!;
        my @BBHEADDUMP = <BBHEADDUMP>;
        close(BBHEADDUMP);

        my $bbexist = 1;
        foreach my $line (@BBHEADDUMP) {
            if (($line =~ /has_no_images/) || ($line =~ /error_not_found/) || ($line =~ /error_forbidden/)) {
                $bbexist = 0;
                last;
            }
        } 

        if ($bbexist == 0) {
            print "downloadAnnotations: Bounding boxes do not exist for $WNID. Moving to the next WNID.\n\n";
        } else {
            print "downloadAnnotations: Downloading bounding boxes for $WNID...\n";
            $BBOX_URL2 =~ s/\[wnid\]/$WNID/;
            if ($use_proxy) {
                system("curl -x \"$PROXY_URL\" \"$BBOX_URL2\" -# -f --cookie-jar $TMP_DIR/cookies -A \"$USER_AGENT\" -o \"$destDir/$WNID-BB.tar.gz\"");
            } else {
                system("curl \"$BBOX_URL2\" -# -f --cookie-jar $TMP_DIR/cookies -A \"$USER_AGENT\" -o \"$destDir/$WNID-BB.tar.gz\"");
            }
            $BBOX_URL2 =~ s/$WNID/\[wnid\]/;
            print "downloadAnnotations: Done.\n";
        }
    }
    return 1;
}
1;
