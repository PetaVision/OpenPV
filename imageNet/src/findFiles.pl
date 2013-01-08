#!/usr/bin/env perl

############
## findFiles.pl
##
## Written by:
##      Dylan Paiton
##      Los Alamos National Laboratory, Group ISR-2 
##      paiton@lanl.gov
##
## Create a lists of lists of the files in their sub dirs
############
sub findFiles($$) {

    my $path = $_[0];
    my $ext  = $_[1];
    my @allFiles;

    my $escPath = quotemeta($path);

    foreach my $item (glob "$escPath/*") {
        if (-d $item) { #if it is a directory
            my @subFiles = &findFiles($item,$ext);
            foreach my $fil (@subFiles) {
                next if (grep {$_ eq $fil} @allFiles);
                push(@allFiles,$fil);
            }
        } else { #Item is not a directory
            if ($item =~ /\.$ext/i) { #Item is has requested file extension
                push(@allFiles,$item);
            }
        }
    }

    return (@allFiles);
}
1;
