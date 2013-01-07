#!/usr/bin/env perl

############
## getFileExts.pl
##
## Written by:
##      Dylan Paiton
##      Los Alamos National Laboratory, Group ISR-2 
##      paiton@lanl.gov
##
## Returns a list of extensions in given path
############

sub getFileExts($) {

    my $path = $_[0];
    my @allExts;

    my $escPath = quotemeta($path);

    my %seen = ();
    foreach my $item (glob "$escPath/*") {
        if (-d $item) { #if it is a directory
            push(@allExts,&findFiles($item));
        } else { #Item is not a directory
            if ($item =~ /\.([\d\w]+)$/i) { #Get file extension
                next if $seen{$1}++;
                push(@allExts,$1);
            }
        }
    }
    return @allExts;
}
1;
