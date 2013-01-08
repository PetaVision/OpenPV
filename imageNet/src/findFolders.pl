#!/usr/bin/env perl

############
## findFolders.pl
##
## Written by:
##      Dylan Paiton
##      Los Alamos National Laboratory, Group ISR-2 
##      paiton@lanl.gov
##
## Create a lists of lists of the folders in their sub dirs
############
sub findFolders($$) {

    my $path = $_[0];
    my $ext  = $_[1];
    my @allFolders;

    my $escPath = quotemeta($path);

    foreach my $item (glob "$escPath/*") {
        if (-d $item) { #if it is a directory
            if ($item =~ m/n[\d]+/) {
                unless (grep {$_ eq $item} @allFolders) {
                    push(@allFolders,$item);
                }
            }
            my @subFolders = &findFolders($item,$ext);
            foreach my $fol (@subFolders) {
                next unless (-d $fol); #Only continue if the subFolder is actually a folder
                if ($fol =~ m/n[\d]+/) {
                    unless (grep {$_ eq $item} @allFolders) {
                        push(@allFolders,$fol);
                    }
                }
            }
        }
    }

    return (@allFolders);
}
1;
