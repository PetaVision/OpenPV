#!/usr/bin/env perl

#Set up output dir
sub makeTempDir {
    use globalVars;
    my $TMP_DIR = getTempDir globalVars();

    unless (-d $TMP_DIR) {
        system("mkdir -p $TMP_DIR") and die "makeTempDir: Couldn't make temp directory: $TMP_DIR\n\tError: $!\n";
    }

    return $TMP_DIR;
}
1;
