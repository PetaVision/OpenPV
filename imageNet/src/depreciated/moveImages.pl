#!/usr/bin/env perl

############
## moveImages.pl
##
## Written by:
##      Dylan Paiton
##      Los Alamos National Laboratory, Group ISR-2 
##      paiton@lanl.gov
##
## There are 3 main modes of opperation:
##  1] &moveImages($root_dir, $category)
##      move a category ($category) and all of its children to the designated directory ($root_dir)
##
##  2] &moveImages($root_dir,$category,$num)
##      move the top $num categories in the subfolder $IMG_DIR/$category ($IMG_DIR is set within countImages.pl) 
##
##  3] &moveImages($root_dir,$category="rt",$num)
##      subset of 2], finds the top $num categories in the folder $IMG_DIR/ directory and moves those to $root_dir
##
##  It would be nice to have a 4th option which is a subset of 1] and moves the top children from a category, instead of all of them.
## 
## TODO:
##  For mode 1] - recognize if parents of previous $child are the same as current $child
##      If so: don't prompt user to enter parents, but use assumed parents
##      If not: prompt user to enter parents
##      *** I have the beginnings of this working, although it is not as clean as I would like
##
##  As of now, modes 2 and 3 do not work
##
############

require 'findChildren.pl';
require 'countImages.pl';
require 'findParents.pl';
require 'getPath.pl';
require 'extractImages.pl';

unless (scalar(@ARGV) >= 2) {
    die "Usage: ./moveImages.pl \"root_dir\" \"category\" [\"num\"]\n\n";
}

$root_dir = $ARGV[0];
chomp($root_dir);
$root_dir =~ s/\/$//;
#$root_dir = "~/Documents/Work/LANL/PANN/neural_computing/data/imagenet/pool";
my $category = $ARGV[1];
if ($ARGV[2]) {
	$num = $ARGV[2];
}

#countImages has a caching option, so that you dont have to recount all images.
#Set this variable to "y" to use cache.
$use_cache = "y";

$currDir = `pwd`;
chomp($currDir);
$TMP_DIR = "$currDir/../tmp";
$IMG_DIR="$currDir/../../archivedImages";

unless (-d $TMP_DIR) {
    system("mkdir -p $TMP_DIR");
}
unless (-d $root_dir) {
    system("mkdir -p $root_dir");
}

$esccurrDir = quotemeta($currDir);
$esccurrDir =~ s/\\\//\//g;

#If we are given $num, assume mode 2] or 3]. Else, assume mode 1]
if ($num) {
    print "\n\nMoving top $num sub-categories in \"$category\" to the \"$root_dir\" directory...\n";

    if ($use_cache =~ /y/) {
        unless (-e "$TMP_DIR/cache.cch") {
            print "\nCache file not found; proceeding without cache.\n";
            $use_cache = "n";
        }
    }

    if ($use_cache =~ /y/) {
#Read in categories from cache, pre-sorted according to amount of images/category
        open(CACHE_READ,"<", "$TMP_DIR/cache.cch") or die "\n\nFAILED to open the cache file.\n$!\n\n";
        @CACHE_READ = <CACHE_READ>;
        close(CACHE_READ);

#Grab top $num categories
        if (scalar(@CACHE_READ) > $num) {
            for($i=0; $i<(scalar(@CACHE_READ)-$num); $i+=1) {
                $pop = pop(@CACHE_READ);
            }
            @CATEGORIES = @CACHE_READ
        } else {
            print "Cache file contains less than $num categories; proceeding without cache.\n";
            @CATEGORIES = countImages($category, $num);
        }
    } elsif ($use_cache =~ /n/) {
#if not using cache, count all categories
        @CATEGORIES = countImages($category, $num);
    } else {
        die "\n\nERROR: Invalid input for 'use_cache'. Please enter 'y' or 'n'\n\n";
    }

    my $escIMG_DIR = quotemeta($IMG_DIR);
    foreach my $category (@CATEGORIES) {
        chomp($category);
        print "\n\n=====================\nMoving images from \"$category\" and its children to the \"$root_dir\" folder...";
        foreach $subcat (glob "$escIMG_DIR/$category/*") {
            $subcat =~ s/$IMG_DIR\/$category\///;
            @CHILDREN = findChildren($subcat);
            foreach my $child (@CHILDREN) {
                chomp($child);
                print "\n\n=====================\nMoving \"$child\" images...\n";
                doMove($child);
            }
        }
    }
} else {
    print "\n\nMoving images from \"$category\" and its children to the \"$root_dir\" folder...\n";
    @CHILDREN = findChildren($category);
    foreach my $child (@CHILDREN) {
        chomp($child);
        print "\n\n=====================\nMoving \"$child\" images...\n";
        doMove($child);
    }
}

if ($result == 1) {
    print "Program Complete.\n";
} else {
    print "Program Failed.\n";
}

sub doMove {
    $cat = $_[0];
    my @PARENTS = undef;
    @PARENTS = findParents($cat, @PREVPARENTS);
    @PREVPARENTS = @PARENTS;
    $cat = $PARENTS[0];
    system("cp $TMP_DIR/folderStructure.xml $root_dir/folderStructure.xml");
    $dest_dir = getPath($root_dir,$cat);
    $result = extractImages($dest_dir, $cat);
    #if ($result == 0) {
    #    die;
    #}
}
1;
