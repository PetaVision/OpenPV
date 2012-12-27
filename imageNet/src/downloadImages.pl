#!/usr/bin/env perl

############
## downloadImages.pl
##
## Written by:
##      Dylan Paiton
##      Los Alamos National Laboratory, Group ISR-3 
##      paiton@lanl.gov
## 
## Extracts $inputimages from $ARCH_DIR to $destDir
##  Inputs:
##      $inputCategory = desired category synset ID (can be ID, text file, or folder)
##      $inputDir      = directory containing image archive files
##      $destDir       = desired output directory
##          *Note: There is no need to escape spaces for $destDir.
############

#####
##Uncomment below to run from command line
##This must stay commented in order to call this function from another program
if ($ARGV[0] && $ARGV[1]) {
    $out = &downloadImages($ARGV[0], $ARGV[1]);
    print "downloadImages: Program complete.\n";
} else {
    &diPostUsage();
}
#####

sub diPostUsage () {
    die "\n\nUsage: ./downloadImages.pl WNID destination_dir\n\tWNID can be a text file, a folder path, or a single ID\n\n\n";
}

sub downloadImages ($$) {
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

    #REPLACE THESE WITH THE APPROPRIATE USER AND KEY
    #####
    my $USER_NAME  = "user";
    my $ACCESS_KEY = "key";
    #####
    my $USER_AGENT = "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)";
    my $IMG_URL    = "http://www.image-net.org/download/synset?wnid=[wnid]&username=[username]&accesskey=[accesskey]&release=latest";

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
    print "\ndownloadImages: Would you like to download the child synset images? [y/n] ";
    my $userChoice = <STDIN>;
    chomp($userChoice);
    my $correctAnswer = 0;
    while ($correctAnswer == 0) {
        if (($userChoice =~ m/^y$/) || ($userChoice =~ m/^n$/)) {
            $correctAnswer = 1;
            last;
        }
        print "downloadImages: Please respond with 'y' or 'n': ";
        $userChoice = <STDIN>;
        chomp($userChoice);
    }
    if ($userChoice =~ /y/) {
        print "downloadImages: Downloading child synset annotations.\n";
        $getChildren = 'y';
    } else {
        print "downloadImages: Not downloading child synset annotations.\n";
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
        print "downloadImages: Getting list of files from the input dir $inputCategory\n";
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
        print "downloadImages: Found $numFilesFound WNIDs to find images for.\n";

        @inArray = @fileList;
    } else { #shouldn't happen
        &eiPostUsage();
    }

    #Get children of each WNID if needed
    if ($getChildren =~ /y/) {
        print "downloadImages: Adding children to download list.\n";
        my @kidsArray;
        foreach my $WNID (@inArray) {
            my ($namesRef, $idsRef) = &listChildren($ARGV[0]);
            my @names = @$namesRef;
            my @WNIDs = @$idsRef;
            push(@kidsArray,@WNIDs);
        }
        push (@inArray,@kidsArray);

        my $numFilesFound = scalar(@inArray);
        print "downloadImages: Found $numFilesFound WNIDs to find images for.\n";
    }

    my %seen = ();
#Parse through input items, everything inside this loop is done for each input item
    foreach my $WNID (@inArray) {
        next if $seen{$WNID}++; #Array probably has duplicates

        print "\n-------------------\n";
        chomp($WNID);

        $IMG_URL =~ s/\[wnid\]/$WNID/;
        $IMG_URL =~ s/\[username\]/$USER_NAME/;
        $IMG_URL =~ s/\[accesskey\]/$ACCESS_KEY/;

        #Check to make sure image tar file exists
        if ($use_proxy) {
            system("curl -x \"$PROXY_URL\" \"$IMG_URL\" -s --cookie $TMP_DIR/cookies --cookie-jar $TMP_DIR/cookies -I 1> $TMP_DIR/headdump.log");
        } else {
            system("curl \"$IMG_URL\" -s --cookie $TMP_DIR/cookies --cookie-jar $TMP_DIR/cookies -I 1> $TMP_DIR/headdump.log");
        }
        open(HEADDUMP,"<","$TMP_DIR/headdump.log") or die "Could not open $TMP_DIR/headdump.log!\nERROR: $!\n";
        @HEADDUMP = <HEADDUMP>;
        close(HEADDUMP);
        $imgexist = 1;
        foreach $line(@HEADDUMP) {
            if (($line =~ /has_no_images/) || ($line =~ /error_not_found/) || ($line =~ /error_forbidden/)) {
                $imgexist = 0;
                print "downloadImages: The image group does not exist for $WNID. Moving to the next WNID.\n\n";
                last;
            }
        }
        #Continue if tar has already been downloaded
        unless ($imgexist == 1) {
            next;
        }

        #Check to see if the tar file has already been downloaded
        if (-e "$destDir/$WNID.tar") {
            print "downloadImages: The tar file for $WNID already exists in $destDir. Moving to the next WNID.\n\n";
            last;
        }

        #Download image tar file
        print "downloadImages: Downloading $WNID.tar\n";
        
        if ($use_proxy) {
            system("curl -# -x \"$PROXY_URL\" \"$IMG_URL\" -o \"$destDir/$WNID.tar\" -w \"Downloaded %{size_download} bytes in %{time_total} seconds.\" --connect-timeout 120 > $TMP_DIR/downdat.log");
        } else {
            system("curl -# \"$IMG_URL\" -o \"$destDir/$WNID.tar\" -w \"Downloaded %{size_download} bytes in %{time_total} seconds.\" --connect-timeout 120 > $TMP_DIR/downdat.log");
        }
        #Print how much was downloaded
        if (-e "$destDir/$WNID.tar") {
            open(DOWNDAT,"<","$TMP_DIR/downdat.log") or die $!;
            @DOWNDAT = <DOWNDAT>;
            close(DOWNDAT);

            foreach $line(@DOWNDAT) {
                if ($line =~ /Downloaded ([\d]+) bytes in ([\d\.]+) seconds./) {
                    $down_amnt = $1;
                    $time = $2;
                    last;
                } else {
                    $down_amnt = 0;
                    $time = 0;
                }
            }
            print "downloadImages: Downloaded $down_amnt bytes in $time seconds.\n";
        } else {
            print "\n\nERROR: DOWNLOAD FAILED...\nWNID: $WNID\n\n";
            next;
        }

        $IMG_URL =~ s/$WNID/\[wnid\]/;
        $IMG_URL =~ s/$USER_NAME/\[username\]/;
        $IMG_URL =~ s/$ACCESS_KEY/\[accesskey\]/;
    }
    return 1;
}
1;
