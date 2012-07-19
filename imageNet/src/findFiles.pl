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
    my (@allFiles, @subFiles);

    my $escPath = quotemeta($path);

    foreach $item (glob "$escPath/*") {
        if (-d $item) { #if it is a directory
            push(@allFiles,&findFiles($item,$ext));
        } else { #Item is not a directory
            if ($item =~ /\.$ext/i) { #Item is has requested file extension
                push(@allFiles,$item);
            }
        }
    }

    return (@allFiles);
}
1;
