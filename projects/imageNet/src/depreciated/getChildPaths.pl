#!/usr/bin/env perl

############
## getChildPaths.pl
##
## Written by:
##      Dylan Paiton
##      Los Alamos National Laboratory, Group ISR-2 
##      paiton@lanl.gov
## 
## Finds the paths to $category and all of its sub-categories, located within $root_dir. 
## Prints them to $root_dir/paths.txt
## An xml file called folderStructure.xml which contains all of the hierarchy information must be located in $root_dir
##
##  Inputs:
##      $root_dir = desired output directory
##          *Note: There is no need to escape spaces for $root_dir.
##      $category = desired category
## 
## TODO:
##   
##
############

##Uncomment below to run from command line
##This must stay commented in order to call this function from another program
#if ($ARGV[0] && $ARGV[1]) {
#    $out = &getChildPaths($ARGV[0], $ARGV[1]);
#} else {
#    die "Usage: ./findParents.pl \"category\" \"root_directory\"\n";
#}

sub getChildPaths {
    require 'findChildren.pl';
    require 'getPath.pl';
    
    my $root_dir = $_[0];
    my $category = $_[1];

    $root_dir =~ s/\s/\\ /g;
    $root_dir =~ s/\,/\\\,/g;
    $root_dir =~ s/\'/\\\'/g;

    open(OUTFILE,">","$root_dir/paths.txt");
    @CHILDREN = findChildren($category);
    foreach my $child (@CHILDREN) {
        my $location = getPath($root_dir, $child);
        print OUTFILE "${location}/images\n";
    }
    close OUTFILE;

    print "\nScript complete.\n\n";
}
