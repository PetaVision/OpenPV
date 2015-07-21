#!/usr/bin/env perl

############
## countimages.pl
##
## Written by:
##      Dylan Paiton
##      Los Alamos National Laboratory, Group ISR-2 
##      paiton@lanl.gov
##
## Count the images within the $IMG_DIR folder
##   uses tar -t to list tar contents
##   Inputs:
##       $num_cats is the number of most populated categories you wish to return
##       $cat is which category you want to count the subcategories of. If $cat = 'rt' then count all of the parent categories.
##           a 'category' is defined as a folder within $IMG_DIR.
##
## TODO:
##   catch any errors from tar execution
##       print file name when error occurs
##
############


##Uncomment below to run from command line
##This must stay commented in order to call this function from another program
#unless (scalar(@ARGV) >= 2) {
#    die "Usage: ./countImages.pl \"num_cats_to_return\" \"category\"\n\n";
#}
#if ($ARGV[0] && $ARGV[1]) {
#    @OUTPUT = &countImages($ARGV[0],$ARGV[1]);
#    print "Categories in order:\n";
#    print join("\n",@OUTPUT), "\n";
#} else {
#    die "Usage: ./countImages.pl \"num_cats_to_return\" \"category\"\n\n";
#}

sub countImages {

    my $cat = $_[0];
    my $num_cats = $_[1];
    chomp($cat);

    undef @OUT;
    
    my $currDir = `pwd`;
    chomp($currDir);
    my $TMP_DIR = "$currDir/../tmp";
    my $IMG_DIR="$currDir/../../archivedImages";

    unless (-d $TMP_DIR) {
        system("mkdir -p $TMP_DIR");
    }

    my $all = "";

    if ($cat =~ /rt/) {
        $all = "y";
    } else {
        $all = "n";
#Category ($cat) must match a folder in the ./$IMG_DIR directory.
        if (-e "$IMG_DIR/$cat") {
            $IMG_DIR .= "/$cat";
        } else {
            die "\nInvalid category. \"$cat\" does not exist in \"$IMG_DIR\"\n";
        }
    }

    print "Counting images in $IMG_DIR...\n";

    @OUT = &doCount($IMG_DIR, $TMP_DIR, $all);

    print "Returning sorted categories.\n";
    return @OUT;
}

sub doCount {
    my $IMG_DIR = $_[0];
    my $TMP_DIR = $_[1];
    my $all = $_[2];
    my ($totcount_cat, $count, $catcount, $totcount) = 0;
    undef @SORT_NAMES;

    my $escIMG_DIR = $IMG_DIR;
    $escIMG_DIR =~ s/\s/\\ /g;

#Count the images using glob to get to the directory and tar -t to get the image list
    foreach $dir (glob "$escIMG_DIR/*") {
        print "\n$dir\n";
        $escdir = quotemeta($dir);

        if ($all =~ /y/) {
            foreach $sub_dir (glob "$escdir/*") {
                $escsubdir = quotemeta($sub_dir);
                foreach $file (glob "$escsubdir/*.tar") {
                   $escfile = quotemeta($file);
                   $count += `tar -tf $escfile | wc -l`;
                }
                $totcount += $count;
                $totcount_cat += $count;
                $count = 0;
                print ".";
            }
        } elsif ($all =~ /n/) {
            foreach $file (glob "$escdir/*.tar") {
                $escfile = quotemeta($file);
                $count += `tar -tf $escfile | wc -l`;
            }
            $totcount += $count;
            $totcount_cat += $count;
            print "$count ";
            $count = 0;
        } else {
            die "\n\nERROR: $all is not 'y' or 'n'\n\n";
        }

        $escimgdir = quotemeta($IMG_DIR);
        $dir =~ s/\\ / /g;
        $dir =~ s/$escimgdir\///;
        $dir =~ s/\/[n\d]+//;
        push(@NAMES,$dir);
        push(@COUNTS,"$totcount_cat". "." ."$catcount"); 
        $totcount_cat = 0;
        $catcount += 1;
    }
    print "\nDone. Sorting categories...\n"; 

#Sort counts and push the top $num_cats to an array for output
    @COUNTS = sort {$b <=> $a} @COUNTS;
	
    $maxcatcount = $COUNTS[0];
    $maxcatcount =~ s/\.^[\d]+//;
    for ($i=0; $i<scalar(@COUNTS); $i+=1) {
        $COUNTS[$i] =~ s/^[\d]+\.//;
        push(@SORT_NAMES,@NAMES[$COUNTS[$i]]);
    }
    undef @NAMES;
    undef @COUNTS;

	print "Done.\n";

	unless (-e "$TMP_DIR") {
        system("mkdir -p $TMP_DIR");
    }
	open(CACHE_WRITE,">","$TMP_DIR/cache.cch") or die "\n\nFAILED to open the cache file\n$!\n\n";
	foreach my $item (@SORT_NAMES) {
		print CACHE_WRITE "$item\n";
	}
	close CACHE_WRITE;

    print "Final image count is $totcount\nFinal category count is $catcount\n";
    print "Most populated category is $SORT_NAMES[0] with $maxcatcount images.\n";

    if (scalar(@SORT_NAMES) > $num_cats) {
        for($i=0; $i<(scalar(@SORT_NAMES)-$num_cats); $i+=1) {
            $pop = pop(@SORT_NAMES);
        }
    }

    return @SORT_NAMES;
}
1;
