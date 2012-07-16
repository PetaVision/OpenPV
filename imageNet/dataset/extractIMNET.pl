#!/usr/bin/env perl

############
## extractIMNET.pl
##
## Written by:
##      Dylan Paiton
##      Los Alamos National Laboratory, Group ISR-3 
##      paiton@lanl.gov
## 
## Extracts $inputimages from $ARCH_DIR to $destDir
##  Inputs:
##      $inputCategory = desired category synset ID
##      $destDir   = desired output directory
##          *Note: There is no need to escape spaces for $destDir.
############

require 'listChildren.pl';
require 'listParents.pl';
require 'findFiles.pl';

#####
##Uncomment below to run from command line
##This must stay commented in order to call this function from another program
if ($ARGV[0] && $ARGV[1] && $ARGV[2]) {
    $out = &extractIMNET($ARGV[0], $ARGV[1], $ARGV[2]);
    print "extractIMNET: Program complete.\n";
} else {
    die "\nUsage: ./extractIMNET.pl WNID input_dir destination_dir\n\tWNID can be a text file, a folder path, or a single ID\n";
}
#####

sub extractIMNET ($$$){
    use warnings;

    my $inputCategory = $_[0];
    chomp($inputCategory);
    my $inputDir  = $_[1];
    chomp($inputDir);
    my $destDir   = $_[2];
    chomp($destDir);

#Set up temp dir
    my $currDir = `pwd`;
    chomp($currDir);
    $currDir =~ s/\s/\\ /g;
    my $TMP_DIR = "$currDir/temp";
    unless (-d $TMP_DIR) {
        system("mkdir -p $TMP_DIR");
    }

#Clean up category
    $inputCategory =~ s/\s/\\ /g;

#Clean up input (arch) dir 
    $inputDir =~ s/\/$//g;
    $inputDir =~ s/\s/\\ /g;

#Clean up dest dir
    $destDir =~ s/\/$//g;
    $destDir =~ s/\s/\\ /g;

#Ask user if the program should get the child nodes
    print "\nextractIMNET: Would you like to extract the children of the input? [y/n] ";
    my $userChoice = <STDIN>;
    chomp($userChoice);
    my $correctAnswer = 0;
    while ($correctAnswer == 0) {
        if (($userChoice =~ m/^y$/) || ($userChoice =~ m/^n$/)) {
            $correctAnswer = 1;
            last;
        }
        print "extractIMNET: Please respond with 'y' or 'n': ";
        $userChoice = <STDIN>;
        chomp($userChoice);
    }

#Check to see if the user has input a WNID, a folder, or a text file listing WNIDs
    my $inputType = 0; #1=WNID, 2=folder, 3=file
    if ($inputCategory =~ m/^n\d+$/) {
        $inputType = 1;
    } elsif (($inputCategory =~ m/\.txt$/) || ($inputCategory =~ m/\.html$/)) {
        $inputType = 2;
    } elsif (-d $inputCategory) {
        $inputCategory =~ s/\/$//g;
        $inputCategory =~ s/\s/\\ /g;
        $inputType = 3;
    } else {
        die "\nUsage: ./extractIMNET.pl WNID input_dir destination_dir\n\tWNID can be a text file, a folder path, or a single ID\n";
    }

    if ($inputType==1) { #The user has input a synset id
        @inArray = ($inputCategory);
    } elsif ($inputType==2) { #The user has input a text file full of synset ids
        #Open input file
        open(INTEXT,"<", $inputCategory) or die $!;
        @inArray = <INTEXT>;
        close(INTEXT);
    } elsif ($inputType==3) { #The user has input a dir full of tar files
        print "extractIMNET: Getting list of files from the input dir $inputCategory\n";
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
        print "extractIMNET: Found $numFilesFound files to extract!\n";

        @inArray = @fileList;
    } else { #shouldn't happen
        die "\nUsage: ./extractIMNET.pl WNID input_dir destination_dir\n\tWNID can be a text file, a folder path, or a single ID\n";
    }

#Parse through input items, everything inside this loop is done for each input item
    print "\n-------------------\n";
    foreach my $lineInput (@inArray) {

        my $result = 0;
        if ($userChoice =~ /^y$/) { #Grab children
            my $HYPONYM_URL="http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid=[wnid]&full=1";

            print "\nextractIMNET: Downloading list of child synsets...\n";
            $HYPONYM_URL =~ s/\[wnid\]/$lineInput/;
            system("curl \"$HYPONYM_URL\" -# --cookie $TMP_DIR/cookies > $TMP_DIR/child_synsets.txt");  
            $HYPONYM_URL =~ s/$lineInput/\[wnid\]/;
            print "extractIMNET: Done.\n";

            open(SYNSETS,"<","$TMP_DIR/child_synsets.txt") or die "Could not open $TMP_DIR/child_synsets.txt.\nERROR: $!\n";
            my @synsets= <SYNSETS>;
            close(SYNSETS);

            foreach my $WNID (@synsets) {
                chomp($WNID);
                $WNID =~ s/\R//g; #Remove new-line
                $WNID =~ s/^\-//g; #Remove - before WNID
                $result = &doImageExtraction($WNID,$inputDir,$destDir);
                if ($result == 1) {
                    print "\n\n-------------------\n\n";
                } else {
                    print "extractIMNET: Failed to extract $WNID.\n\n-------------------\n\n";
                }
            }
        } else {
            
           $result = &doImageExtraction($lineInput,$inputDir,$destDir);
           if ($result == 1) {
               print "extractIMNET: Extracted $lineInput!\n\n-------------------\n\n";
           } else {
               print "extractIMNET: Failed to extract $lineInput\n";
           }
        }
    }
    return 1;
}

sub doImageExtraction ($$$) {
############
## doImageExtraction
##
## Extracts $categoryWNID images from $ARCH_DIR to $destDir
##  Inputs:
##      $categoryWNID = desired category synset ID
##      $destDir = desired output directory
##          *Note: There is no need to escape spaces for $destDir.
############

#Format input
    my $categoryWNID = $_[0];
    chomp($categoryWNID);
    my $inputDir = $_[1];
    chomp($inputDir);
    my $destDir = $_[2];
    chomp($destDir);

    print "doImageExtraction: Looking for $categoryWNID to put into $destDir\n";

#Set up temp dir
    my $currDir = `pwd`;
    chomp($currDir);
    $currDir =~ s/\s/\\ /g;
    my $TMP_DIR = "$currDir/temp";
    unless (-d $TMP_DIR) {
        system("mkdir -p $TMP_DIR");
    }


#Verify that the tar file exists in the archived image directory
    my $ARCH_DIR=$inputDir;
    opendir(DIR,$ARCH_DIR) or die "doImageExtraction: Can't open directory $ARCH_DIR!\nError: $!\n";
    my @fileList = grep !/^\.\.?/,readdir(DIR);
    closedir(DIR);
    my $found    = 0;
    my $archFile = "";
    foreach my $file (@fileList) {
       if ($file =~ m/$categoryWNID/) {
           $archFile = $file;
           $found = 1;
           last;
       }
    }
    unless ($found) {
        print "doImageExtraction: Couldn't find the synset \"$categoryWNID\" in \"$ARCH_DIR\"\n",
            "The contents of the searched directory are:\n",
            join("\n",@fileList),"\n\n";
        return 0;
    }


#Set up directory list
    my ($namesRef, $idsRef) = listParents($categoryWNID);

    my @parents  = reverse(@$namesRef); #Change to @$idsRef in order to extract to folders with the synset ID

    $categoryName = $parents[-1]; #Last element in array should be the input category


    undef @outputDirs;
    $outputDir = $destDir;

    foreach my $parent (@parents) {
        chomp($parent);
        $sub_dir = $parent;
        $sub_dir =~ s/\s/\\ /g;
        $sub_dir =~ s/\,/\\\,/g;
        $sub_dir =~ s/\'/\\\'/g;
        $sub_dir =~ s/\//\\\_/g;
        $outputDir = $outputDir . "/" .  $sub_dir;
        if ($parent =~ m/$categoryName/) {
            push(@outputDirs,$outputDir);
            $outputDir = $destDir;
        }
    }

#Create directories & extract files
    print "doImageExtraction: Extracting images...\n\n";
    my $firstRun = 1; 
    my $firstDir = '';
    foreach my $dir (@outputDirs) {
        chomp($dir);

        my $cleanDir = $dir;
        $cleanDir =~ s/\\//g;
        system("mkdir -p $dir") unless (-d $cleanDir);
        die "Failed to make $cleanDir!\n\n" unless (-d $cleanDir);

        if ($firstRun) {
            print "doImageExtraction:\n\tCategory:\t$categoryWNID\n\tArchive file:\t$ARCH_DIR/$archFile\n\tDestination:\t$cleanDir/images\n\n";

            system("mkdir $dir/images") unless (-d "$cleanDir/images");
            system("tar -xkzf $ARCH_DIR/$archFile -C $dir/images/ 2> $TMP_DIR/tarout.err");
            #system("rm -f $destDir/images/synset\_info.txt");

            open(TARERR,"<","$TMP_DIR/tarout.err") or die("Could not open $TMP_DIR/tarout.err\nERROR: $!\n");
            my @TARERR=<TARERR>;
            close(TARERR); 
            foreach my $line (@TARERR) {
                if ($line =~ /Already exists/) {
                    print "The images have already been extracted for this tar file.\n";
                    return 0;
                } elsif (($line =~ /Failed to open/) || ($line =~ /Error/) || ($line =~ /could not chdir/)) {
                    print "doImageExtraction: WARNING: Tar extraction failed for file $archFile.\n  $line\n\n";
                    return 0;
                }
            }
            undef @TARERR;

            $firstRun = 0;
            $firstDir = "$dir/images";
        } else {
            system("ln -s $firstDir $dir/images");
        }
    }

    return 1;
}
1;
