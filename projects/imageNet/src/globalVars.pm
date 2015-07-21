#!/usr/bin/env perl

package globalVars;
    sub getUseProxy {
        my $useProxy = 0;
        return $useProxy;
    }
    sub getProxyURL {
        return "http://proxyout.lanl.gov:8080/";
    }
    sub getTempDir {
        my $currDir = `pwd`;
        chomp($currDir);
        $currDir =~ s/\s/\\ /g;
        my $TMP_DIR = "$currDir/temp";

        return $TMP_DIR;
    }
1;
