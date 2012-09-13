#!/usr/bin/env perl

############
## downloadImages.pl
##
## Written by:
##      Dylan Paiton
##      Los Alamos National Laboratory, Group ISR-2 
##      paiton@lanl.gov
## 
## Download images from http://www.image-net.org/ and sort into folders.
## For use at the Synthetic Cognition Lab under supervision of Dr. Steven P. Brumby only.
## Do not distribute this program.
## 
## TODO:
##  This code could use a serious cleanup
##
##  Add option to skip or overwrite if tar file already exists
##      right now it just skips it if it has already been downloaded
##      Maybe compare old/new tar sizes and only download if new > old?
##
##  escape key for getting out of downloads
##      sub INT_handler {
##          #mark location in list
##          #mark download location, need to resume later
##          #post error message
##          exit(0);
##      }
##      $SIG{'INT'} = 'INT_handler';
##
##  increase accuracy of $img_down
##
##  consider using Image-Net XML structure for naming files, so that they are named appropriately
##      As it stands, the naming scheme has some letter capitalization issues
##      It takes much more time to use the image-net xml structure than the current method because of the size of the file
##      Either remove cap dependancies in subsequent perl scripts, or make a decision about which to use
##
##  Add a "use proxy" flag, and modify all curl executions to either use proxy or not
############

#use warnings;

##Setting $use_proxy to 1 is required in order to download images from within LANL's yellow network.
my $use_proxy = 0;

$currDir = `pwd`;
chomp($currDir);
$TMP_DIR="$currDir/../tmp";
$IMG_DIR="$currDir/../../archivedImages";


if ($use_proxy) {
    $PROXY_URL="http://proxyout.lanl.gov:8080/";
}

$USER_NAME="name";
$ACCESS_KEY="key";

$USER_AGENT= "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)";
$WNID_URL="http://www.image-net.org/download/synset?wnid=[wnid]&username=[username]&accesskey=[accesskey]&release=latest";
$SEARCH_URL="http://www.image-net.org/search?q="; 
$IMGNET_URL="http://www.image-net.org/synset?wnid=";
$IMGLIST_URL="http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=";
$BBOX_URL1="http://www.image-net.org/api/download/imagenet.bbox.synset?wnid=";
$BBOX_URL2="http://www.image-net.org/downloads/bbox/[wnid].tar.gz";
$HYPONYM_URL="http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid=[wnid]&full=1";
$NAME_URL = "http://image-net.org/__viz/getControlDetails.php?wnid=[wnid]";

unless (scalar(@ARGV) == 1) {
    die "Usage: ./downloadImages.pl \"search_terms.txt\"\n";
} 

$WNID_URL =~ s/\[username\]/$USER_NAME/;
$WNID_URL =~ s/\[accesskey\]/$ACCESS_KEY/;

##User input required initially
$correct = 0;
print "\n----------------------------------------------" . 
    "\nWould you like to download all of the child synsets? [n] ";
$children = lc(<STDIN>);
until ($correct) {
    if ($children) {
        chomp($children);
        if ($children =~ /^[y]$/) {
            print "Including all child synsets.\n\n";
            $correct = 1;
        } elsif ($children =~ /^[n]?$/) {
            print "Not including all child synsets.\n\n";
            $children = "n";
            until ($correct) {
                print "How many images of each search term would you like to download? [1000] ";
                $img_num = <STDIN>;
                chomp($img_num);
                if ($img_num =~ m/^[\d]+$/) {
                    print "Downloading approximately $img_num images of each synset.\n";
                    print "Please note that the exact number of images downloaded can not be known and an over estimation will be made.\n\n";
                    $correct = 1;
                } else {
                    $img_num = 1000;
                    $correct = 1;
                    print "Using default value, $img_num\n";
                }
            }
        } else {
            print "Please answer 'y' or 'n'\n";
            $children = lc(<STDIN>);
        } 
    } else {
        $children = "n";
    }
}

##Set up directories
print "\nSetting up directories...\n";
system("mkdir $TMP_DIR");
system("mkdir $IMG_DIR");

##Open input file
open(INPUT,"<", $ARGV[0]) or die $!;
@INPUT = <INPUT>;
close(INPUT);

##Parse through search terms, everything inside this loop is done for each search term
foreach $input(@INPUT) {

##Clear variables
    $WNID = 0;
    $down_amnt = 0;
    undef @SOURCE;
    undef @HEADDUMP;
    undef @SYNSETS;

##If eval stops working (I've heard it has bugs, but I have yet to see any) there is a try/catch routine by CPAN
    eval {  ##Try
        chomp($input);
        $origInput = $input;
##Skip comments
        if ($input =~/#/) {
##Rename input in search_terms.txt to be #input so that it isn't searched for again
            if (-e $ARGV[0]){
                open(OUTFILE,">>","$TMP_DIR/tmp_search.txt") or die $!;

                print OUTFILE "$origInput\n";

                close OUTFILE;
            } else {
                print "\n\nWARNING: search_terms.txt does not exist in this directory..\n\n";
            }
            next;
        }
##Encode input for URL characters
        $input =~ s/\%/\%25/g;
        $input =~ s/\s/\+/g;
        $input =~ s/\"/\%22/g;
        $input =~ s/\</\%3C/g;
        $input =~ s/\>/\%3E/g;
        $input =~ s/\#/\%23/g;
        $input =~ s/\{/\%7B/g;
        $input =~ s/\}/\%7D/g;
        $input =~ s/\|/\%7C/g;
        $input =~ s/\\/\%5C/g;
        $input =~ s/\^/\%5E/g;
        $input =~ s/\~/\%7E/g;
        $input =~ s/\[/\%5B/g;
        $input =~ s/\]/\%5D/g;
        $input =~ s/\,/\%2C/g;
        $input =~ s/\`/\%60/g;

##Download the source file for the search result page, called search_output.log
        print "\n\nSearching ImageNet for $origInput" . "...\n";
        if ($use_proxy) {
            system("curl -x \"$PROXY_URL\" \"$SEARCH_URL$input\" -# --cookie-jar $TMP_DIR/cookies > $TMP_DIR/search_output.log");
        } else {
            system("curl \"$SEARCH_URL$input\" -# --cookie-jar $TMP_DIR/cookies > $TMP_DIR/search_output.log");
        }

##Encode input for file naming
        $input =~ s/\&rank\=depth//g;
        $input =~ s/\+/\\ /g;
        $input =~ s/\%20/\_/g;
        $input =~ s/\%22/\-/g;
        $input =~ s/\%3C/\-/g;
        $input =~ s/\%3E/\-/g;
        $input =~ s/\%23/\#/g;
        $input =~ s/\%7B/\{/g;
        $input =~ s/\%7D/\}/g;
        $input =~ s/\%7C/\-/g;
        $input =~ s/\%5C/\-/g;
        $input =~ s/\%5E/\^/g;
        $input =~ s/\%7E/\-/g;
        $input =~ s/\%5B/\[/g;
        $input =~ s/\%5D/\]/g;
        $input =~ s/\%60/\`/g;
        $input =~ s/\%2C/\,/g;
        $input =~ s/\%25/\-/g;
        $input =~ s/\//\-/g;
        $input =~ s/\?/\-/g;
        $input =~ s/\%/\-/g;
        $input =~ s/\*/\-/g;
        $input =~ s/\:/\-/g;
        $input =~ s/\|/\-/g;
        $input =~ s/\"/\-/g;
        $input =~ s/\</\-/g;
        $input =~ s/\>/\-/g;
        $input =~ s/\=/\-/g;

        open(SOURCE,"<","$TMP_DIR/search_output.log") or die("Could not open search_output.log");
        @SOURCE=<SOURCE>;
        close(SOURCE);

##Look for synset ID number associated with search term.
##This grabs the first synset listed, which should be the most popular result. 
        $error = 0;
        foreach $source(@SOURCE) {
            if ($source =~ /matches 0 synsets/) {
                print "\n\nSearch term $origInput matched 0 synsets. Moving to the next term.\n\n";
                $error = 1;
                last;
            } elsif ($source =~ /wnid\=([\w]+)[^\:]+[^\>]+\>([^\<]+)[^\:]+[^\>]+\>([^\<]+)/) {
                $WNID = $1;
                $SYNSET = $2;
                $DEF = $3;
                print "\n\nSearch group number (WNID) = $WNID\nSynset = $SYNSET\nDefinition = $DEF\n\n";
                last;
            } elsif ($source =~/wnid\=([\w]+)/) {
                $WNID = $1;
                print "\n\nSearch group number (WNID) = $WNID\n\n";
                last;
            }
        }
        if ($error) {
            next;
        }
        unless ($WNID) {
            next;
        }

##Download child synsets
        print "\nDownloading list of child synsets...\n";
        $HYPONYM_URL =~ s/\[wnid\]/$WNID/;
        if ($use_proxy) {
            system("curl -x \"$PROXY_URL\" \"$HYPONYM_URL\" -# --cookie $TMP_DIR/cookies > $TMP_DIR/child_synsets.txt");  
        } else {
            system("curl \"$HYPONYM_URL\" -# --cookie $TMP_DIR/cookies > $TMP_DIR/child_synsets.txt");  
        }
        $HYPONYM_URL =~ s/$WNID/\[wnid\]/;
        print "\n\n";

        open(SYNSETS,"<","$TMP_DIR/child_synsets.txt") or die "Could not open $TMP_DIR/child_synsets.txt.\nERROR: $!\n";
        @SYNSETS = <SYNSETS>;
        close(SYNSETS);

        $childcount = `wc -l $TMP_DIR/child_synsets.txt`;
		chomp($childcount);
		if ($childcount =~ /([\d]+)/) {
			$childcount = $1;
            print "There are $childcount subcategories for $input.\n";
		}

        $totDown = 0;
        $numcatsdown = 0;
        foreach $synsets(@SYNSETS) {
            chomp($synsets);
##Clear variables
            undef @HEADDUMP;
            undef @BBHEADDUMP;
            undef @SRCDMP;
            undef @DOWNDAT;
##Grab WNIDs
            if ($synsets =~ /([\w]+)/) { 
                $WNID = $1;
                chomp($WNID);
            } else {
                next;
            }

##Check to make sure images exist
            $WNID_URL =~ s/\[wnid\]/$WNID/;
            if ($use_proxy) {
                system("curl -x \"$PROXY_URL\" \"$WNID_URL\" -s --cookie $TMP_DIR/cookies --cookie-jar $TMP_DIR/cookies -I 1> $TMP_DIR/headdump.log");
            } else {
                system("curl \"$WNID_URL\" -s --cookie $TMP_DIR/cookies --cookie-jar $TMP_DIR/cookies -I 1> $TMP_DIR/headdump.log");
            }
            $WNID_URL =~ s/$WNID/\[wnid\]/;

            open(HEADDUMP,"<","$TMP_DIR/headdump.log") or die "Could not open $TMP_DIR/headdump.log!\nERROR: $!\n";
            @HEADDUMP = <HEADDUMP>;
            close(HEADDUMP);

            $imgexist = 1;
            foreach $line(@HEADDUMP) {
                if (($line =~ /has_no_images/) || ($line =~ /error_not_found/) || ($line =~ /error_forbidden/)) {
                    $imgexist = 0;
                    print "The image group does not exist for $WNID.\nMoving to the next WNID.\n\n";
                    last;
                }
            }
            unless($imgexist) {
                next;
            }

##Check to see if file has already been downloaded
            $WNIDexists = 0;
            foreach $tar (glob "$IMG_DIR/*/*/*.tar") {
                if ($tar =~ /$WNID/g) {
                    $WNIDexists = 1;
                    $WNIDdir = $tar;
                    $WNIDdir =~ s/\/n[\d]+\.tar//g;
                    chomp($WNIDdir);
                    last;
                }
            }

##Download bounding boxes if directory exists
            if ($WNIDexists) {
                print "$WNID file exists at $WNIDdir.\nNot downloading images.\n";
##Check for bounding boxes
                if (-e "$WNIDdir/$WNID-BB.tar.gz") {
                    print "Bounding boxes have already been downloaded.\nMoving to the next WNID.\n\n";
                    next;
                } else {
##Check to make sure bounding boxes exist
                    if ($use_proxy) {
                        system("curl -x \"$PROXY_URL\" \"$BBOX_URL1$WNID\" -s --cookie $TMP_DIR/cookies --cookie-jar $TMP_DIR/cookies -A \"$USER_AGENT\" > $TMP_DIR/BBout.log");
                        $BBOX_URL2 =~ s/\[wnid\]/$WNID/;
                        system("curl -x \"$PROXY_URL\" \"$BBOX_URL2\" -s --cookie-jar $TMP_DIR/cookies -e \"$BBOX_URL1$WNID\" -I 1> $TMP_DIR/BBheaddump.log");
                        $BBOX_URL2 =~ s/$WNID/\[wnid\]/;
                    } else {
                        system("curl -x \"$PROXY_URL\" \"$BBOX_URL1$WNID\" -s --cookie $TMP_DIR/cookies --cookie-jar $TMP_DIR/cookies -A \"$USER_AGENT\" > $TMP_DIR/BBout.log");
                        $BBOX_URL2 =~ s/\[wnid\]/$WNID/;
                        system("curl -x \"$PROXY_URL\" \"$BBOX_URL2\" -s --cookie-jar $TMP_DIR/cookies -e \"$BBOX_URL1$WNID\" -I 1> $TMP_DIR/BBheaddump.log");
                        $BBOX_URL2 =~ s/$WNID/\[wnid\]/;
                    }

                    open(BBHEADDUMP,"<","$TMP_DIR/BBheaddump.log") or die $!;
                    @BBHEADDUMP = <BBHEADDUMP>;
                    close(BBHEADDUMP);

                    $bbexist = 1;
                    foreach $line(@BBHEADDUMP) {
                        if (($line =~ /has_no_images/) || ($line =~ /error_not_found/) || ($line =~ /error_forbidden/)) {
                            $bbexist = 0;
                            last;
                        }
                    }

##Download bounding boxes
                    if ($bbexist == 0) {
                        print "Bounding boxes do not exist for $WNID.\nMoving to the next WNID.\n\n";
                    } else {
                        print "Downloading bounding boxes for $WNID...\n";
                        $BBOX_URL2 =~ s/\[wnid\]/$WNID/;
                        if ($use_proxy) {
                            system("curl -x \"$PROXY_URL\" \"$BBOX_URL2\" -# -f --cookie-jar $TMP_DIR/cookies -A \"$USER_AGENT\" -e \"$BBOX_URL1$WNID\" -o \"$WNIDdir/$WNID-BB.tar.gz\"");
                        } else {
                            system("curl \"$BBOX_URL2\" -# -f --cookie-jar $TMP_DIR/cookies -A \"$USER_AGENT\" -e \"$BBOX_URL1$WNID\" -o \"$WNIDdir/$WNID-BB.tar.gz\"");
                        }
                        $BBOX_URL2 =~ s/$WNID/\[wnid\]/;
                        print "Bounding box downloaded.\nMoving to the next WNID.\n\n";
                    }
                    next;
                }
            } else {
                if (-e "$IMG_DIR/$input") {
                    print "Directory img/$input exists, downloading file to this directory.\n";
                } else {
                    system("mkdir -p $IMG_DIR/$input");
                    print "Made directory img/$input. Downloading file to this directory.\n";
                }
            }

##Name sub-folders to be significant. Using web page because not all archives have synset info in them.
            $NAME_URL =~ s/\[wnid\]/$WNID/;
            if ($use_proxy) {
                system("curl -x \"$PROXY_URL\" -s $NAME_URL > $TMP_DIR/dump.log");
            } else {
                system("curl -s $NAME_URL > $TMP_DIR/dump.log");
            }
            $NAME_URL =~ s/$WNID/\[wnid\]/;

            open(SRCDMP,"<","$TMP_DIR/dump.log");
            @SRCDMP = <SRCDMP>;
            close(SRCDMP);

            foreach $line (@SRCDMP) {
                chomp($line);
                if ($line =~ /\>([a-z|A-Z|\s|\,|\-]+)\</) {
                    $wname = $1;
                    chomp($wname);
                    last;
                }
                else {
                    $wname = "$WNID";
                }
            }
            $wname =~ s/\//\-/g;
            $wname =~ s/\?/\-/g;
            $wname =~ s/\%/\-/g;
            $wname =~ s/\*/\-/g;
            $wname =~ s/\:/\-/g;
            $wname =~ s/\|/\-/g;
            $wname =~ s/\"/\-/g;
            $wname =~ s/\</\-/g;
            $wname =~ s/\>/\-/g;
            $wname =~ s/\=/\-/g;
            $wname = lcfirst($wname);

            unless (-e "$IMG_DIR/$input/$wname") {
                system("mkdir -p \"$IMG_DIR/$input/$wname\"");
            }

##Download images
            print "Downloading images of $input, $WNID to path ../img/$input/$wname...\n";
            $WNID_URL =~ s/\[wnid\]/$WNID/;
            if ($use_proxy) {
                system("curl -x \"$PROXY_URL\" -# \"$WNID_URL\" -o \"$IMG_DIR/$input/$wname/$WNID.tar\" -w \"Downloaded %{size_download} bytes in %{time_total} seconds.\" --connect-timeout 120 > $TMP_DIR/downdat.log");
            } else {
                system("curl -# \"$WNID_URL\" -o \"$IMG_DIR/$input/$wname/$WNID.tar\" -w \"Downloaded %{size_download} bytes in %{time_total} seconds.\" --connect-timeout 120 > $TMP_DIR/downdat.log");
            }
            $WNID_URL =~ s/$WNID/\[wnid\]/;

##Record how much was downloaded
            if (-e "$IMG_DIR/$input/$wname/$WNID.tar") {
                open(DOWNDAT,"<","$TMP_DIR/downdat.log") or die $!;
                @DOWNDAT = <DOWNDAT>;
                close(DOWNDAT);

                foreach $line(@DOWNDAT) {
                    if ($line =~ /Downloaded ([\d]+) bytes in ([\d\.]+) seconds./) {
                        $down_amnt = $1;
                        $time = $2;
                        last;
                    } else {
                        $down_amnt = 0;
                        $time = 0;
                    }
                }
                print "Downloaded $down_amnt bytes in $time seconds.\n";
            } else {
                print "\n\nERROR: DOWNLOAD FAILED...\nWNID: $WNID\n\n";
                next;
            }

##Download bounding boxes
            print "Downloading bounding boxes (if available)\n";
            if ($use_proxy) {
                system("curl -x \"$PROXY_URL\" \"$BBOX_URL1$WNID\" -s --cookie $TMP_DIR/cookies --cookie-jar $TMP_DIR/cookies -A \"$USER_AGENT\" > $TMP_DIR/BBout.log");
            } else {
                system("curl \"$BBOX_URL1$WNID\" -s --cookie $TMP_DIR/cookies --cookie-jar $TMP_DIR/cookies -A \"$USER_AGENT\" > $TMP_DIR/BBout.log");
            }
##Check to make sure bounding boxes exist
            $BBOX_URL2 =~ s/\[wnid\]/$WNID/;
            if ($use_proxy) {
                system("curl -x \"$PROXY_URL\" \"$BBOX_URL2\" -s --cookie-jar $TMP_DIR/cookies -e \"$BBOX_URL1$WNID\" -I 1> $TMP_DIR/BBheaddump.log");
            } else {
                system("curl \"$BBOX_URL2\" -s --cookie-jar $TMP_DIR/cookies -e \"$BBOX_URL1$WNID\" -I 1> $TMP_DIR/BBheaddump.log");
            }
            open(BBHEADDUMP,"<","$TMP_DIR/BBheaddump.log") or die $!;
            @BBHEADDUMP = <BBHEADDUMP>;
            close(BBHEADDUMP);

            $bbexist = 1;
            foreach $line(@BBHEADDUMP) {
                if (($line =~ /has_no_images/) || ($line =~ /error_not_found/) || ($line =~ /error_forbidden/)) {
                    $bbexist = 0;
                    last;
                }
            }
            if ($bbexist == 0) {
                print "Bounding boxes do not exist for $WNID.\n\n";
            } else {
                if ($use_proxy) {
                    system("curl -x \"$PROXY_URL\" \"$BBOX_URL2\" -# -f --cookie-jar $TMP_DIR/cookies -A \"$USER_AGENT\" -e \"$BBOX_URL1$WNID\" -o \"$IMG_DIR/$input/$wname/$WNID-BB.tar.gz\"");
                } else {
                    system("curl \"$BBOX_URL2\" -# -f --cookie-jar $TMP_DIR/cookies -A \"$USER_AGENT\" -e \"$BBOX_URL1$WNID\" -o \"$IMG_DIR/$input/$wname/$WNID-BB.tar.gz\"");
                }
                print "Bounding box downloaded.\n\n";
            }
            $BBOX_URL2 =~ s/$WNID/\[wnid\]/;

##Test to see if we have downloaded enough images
            $numcatsdown += 1;
            if ($children =~ /^[n]$/) {
                $avgimgsize = 94510;
                $totDown += $down_amnt;
                $img_down = $totDown/$avgimgsize;
                if ($img_down >= $img_num) {
                    $img_down =~ s/^([\d]+)\.[\d]+$/$1/;
                    print "\nYou have downloaded about $img_down images in $numcatsdown categories. Moving on to the next search term.\n";
                    last;
                } else {
                    print "\nYou have downloaded about $img_down images in $numcatsdown categories so far.\n";
                    next;
                }
            } 
        } #end foreach $synset
        print "Finished downloading images of $origInput.\n\n";

##Rename input in search_terms.txt to be #input so that it isn't searched for again
        if (-e $ARGV[0]) {
            open(OUTFILE,">>","$TMP_DIR/tmp_search.txt") or die $!;

            if ($children =~ /^[n]$/) {
                print OUTFILE "#$img_num#$origInput\n";
            } else {
                print OUTFILE "#all#$origInput\n";
            }

            close OUTFILE;
        } else {
            print "\nsearch_terms.txt does not exist in this directory..\n";
        }
        
        1; ##Return value for try/catch
    } or do {
##Catch
        print "\n\nProgram errored unexpectedly on WNID: $WNID.\nError: $@\n\n";
        next;
    };
} #end foreach $input

##Replace old search terms with new
if (-e "$TMP_DIR/tmp_search.txt") {
    system("cp -f $TMP_DIR/tmp_search.txt $ARGV[0]");
    system("rm -f $TMP_DIR/tmp_search.txt");
}

print("\n\nEnd of program.\n\n");  
