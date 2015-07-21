#!/usr/bin/env perl

sub checkInputType($) {
    my $userIn = $_[0];

    my $inputType = 0; #1=WNID, 2=folder, 3=file

    if ($userIn =~ m/^n\d+$/) {
        $inputType = 1;
    } elsif (($userIn =~ m/\.txt$/) || ($userIn =~ m/\.html$/)) {
        $inputType = 2;
    } elsif (-d $userIn) {
        $inputType = 3;
    }

    return $inputType;
}
1;
