#!/usr/bin/env perl

############
## extractImages.pl
##
## Written by:
##      Dylan Paiton
##      Los Alamos National Laboratory, Group ISR-2 
##      paiton@lanl.gov
## 
## Extracts $category images from $IMG_DIR to $destDir
##  Inputs:
##      $category = desired category
##      $destDir = desired output directory
##          *Note: There is no need to escape spaces for $destDir.
## 
## TODO:
##   return proper values.. This returns 1 regardless
##      Update: I've started working on this, but it could probably still be more informative.
##   fix param warning
##
############

#####
##Uncomment below to run from command line
##This must stay commented in order to call this function from another program
if ($ARGV[0] && $ARGV[1]) {
    $out = &extractImages($ARGV[0], $ARGV[1]);
} else {
    die "Usage: ./extractImages.pl \"category\" \"destination_directory\"\n\n";
}
#####

sub extractImages {

#Set up temp dir
    my $currDir = `pwd`;
    chomp($currDir);
    my $currDir =~ s/\s/\\ /g;
    my $TMP_DIR = "$currDir/../tmp";
    unless (-d $TMP_DIR) {
        system("mkdir -p $TMP_DIR");
    }

    my $IMG_DIR="$currDir/../../archivedImages";


    print "\nExtracting images...\n";
    my $destDir = $_[0];
    my $category = $_[1];
    chomp($category);
    $category =~ s/\s/\\ /g;
    my @dir = glob "$IMG_DIR/*/$category";
    print "\nCategory:\t\"$category\"\nCurrent Path:\t@dir[0]\nDestination:\t$destDir/images\n";
    $destDir =~ s/\s/\\ /g;

    my $nocat = 0;
    my $dir = @dir[0];
    if (-d "$dir") {
        $dir =~ s/\s/\\ /g;
        $IMG_DIR = $dir;
    } else {
        print "\n\nWARNING: Category \"$category\" not found in $IMG_DIR!\nNot extracting files.\n\n";
        $nocat = 1;
    }

    unless ($nocat) {
        unless (-e "$destDir/images") {
            system("mkdir -p $destDir/images");
        }
        foreach $file (glob "$IMG_DIR/*.tar") {
            $tarfile = $file;
            $tarfile =~ s/\s/\\ /g;
            $tarfile =~ s/\,/\\\,/g;
            $tarfile =~ s/\'/\\\'/g;
            system("tar -xkzf $tarfile -C $destDir/images/ 2> $TMP_DIR/tarout.err");
            system("rm -f $destDir/images/synset\_info.txt");
           
            open(TARERR,"<","$TMP_DIR/tarout.err") or die("Could not open $TMP_DIR/tarout.err\nERROR: $!\n");
            @TARERR=<TARERR>;
            close(TARERR); 
            foreach $line (@TARERR) {
                if ($line =~ /Already exists/) {
                    print "The images have already been extracted for this tar file.\n";
                    last;
                } elsif (($line =~ /Failed to open/) || ($line =~ /Error/)) {
                    die "\n\nERROR: Tar extraction failed.\n  $line\n\n";
                }
            }
            undef @TARERR;
        }
        $check = glob "$currDir/$destDir/images/*.*";
        if ($check) {
            print "\nFinished extracting images.\n";
            return 1;
        } else {
            print "\n\nWARNING: Image exctraction failed. $destDir/images does not contain files.\n\n";
            return 0;
        }
    }
    return 0;
}
1;
