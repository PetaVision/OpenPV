#!/usr/bin/env perl

############
## extractAnnotations.pl
##
## Written by:
##      Dylan Paiton
##      Los Alamos National Laboratory, Group ISR-2 
##      paiton@lanl.gov
## 
## Extracts $category's bounding boxes from $IMG_DIR to $destDir
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

##Uncomment below to run from command line
##This must stay commented in order to call this function from another program
if ($ARGV[0] && $ARGV[1]) {
    $out = &extractAnnotations($ARGV[0], $ARGV[1]);
} else {
    die "Usage: ./extractImages.pl \"category\" \"destination_directory\"\n\n";
}

sub extractAnnotations {
    use File::Glob ':nocase';

    my $currDir = `pwd`;
    chomp($currDir);
    $currDir =~ s/\s/\\ /g;
    my $TMP_DIR = "$currDir/../tmp";
    my $IMG_DIR="$currDir/../../archivedImages";

    unless (-d $TMP_DIR) {
        system("mkdir -p $TMP_DIR");
    }

    print "\nExtracting annotations...\n";
    my $destDir = $_[0];
    my $category = $_[1];
    $category =~ s/\s/\\ /g;
    chomp($category);
    @dir = glob "$currDir/../img/*/$category";
    print "\nCategory:\t\"$category\"\nCurrent Path:\t@dir[0]\nDestination:\t$destDir\n";
    $destDir =~ s/\s/\\ /g;

    unless (-e "$destDir") {
        system("mkdir -p $destDir");
    }

    $nocat = 0;
    $dir = @dir[0];
    if (-d "$dir") {
        $dir =~ s/\s/\\ /g;
        $IMG_DIR = $dir;
    } else {
        print "\n\nWARNING: Category \"$category\" not found in $currDir/../img/!\nCouldn't find $dir\nNot extracting files.\n\n";
        $nocat = 1;
    }

    unless ($nocat) {
        foreach $file (glob "$IMG_DIR/*.tar.gz") {
            $tarfile = $file;
            $tarfile =~ s/\s/\\ /g;
            $tarfile =~ s/\,/\\\,/g;
            $tarfile =~ s/\'/\\\'/g;
            system("tar -xkzf $tarfile -C $destDir/ 2> $TMP_DIR/tarout.err");
           
            open(TARERR,"<","$TMP_DIR/tarout.err") or die("Could not open $TMP_DIR/tarout.err\nERROR: $!\n");
            @TARERR=<TARERR>;
            close(TARERR); 
            foreach $line (@TARERR) {
                if ($line =~ /Already exists/) {
                    print "The bounding boxes have already been extracted for this tar file.\n";
                    last;
                } elsif (($line =~ /Failed to open/) || ($line =~ /Error/)) {
                    die "\n\nERROR: Tar extraction failed.\n  $line\n\n";
                }
            }
            undef @TARERR;
        }
        $check = glob "$currDir/$destDir/Annotation/*/*.xml";
        if ($check) {
            print "\nFinished extracting bounding boxes.\n";
            return 1;
        } else {
            print "\n\nWARNING: Annotation exctraction failed. $destDir does not contain files.\n\n";
            return 0;
        }
    }
    return 0;
}
1;
