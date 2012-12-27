#!/usr/bin/env perl

############
## listParents.pl
##
## Written by:
##      Dylan Paiton
##      Los Alamos National Laboratory, Group ISR-2 
##      paiton@lanl.gov
##
## Download Image-Net structure
## Locate $input in structure
## Return all parents of $input, along all trees
## Output two semi-colon separated files: listing names & WNIDs
##
## TODO:
##
############

require 'findFiles.pl';
require 'makeTempDir.pl';
require 'checkInputType.pl';

#########################
##Uncomment _below_ to run from command line
##Leave _below_ commented in order to call this function from another program
#########################
#if ($ARGV[0]) {
#    my ($namesRef, $idsRef) = &listParents($ARGV[0]);
#
#    my @names = @$namesRef;
#    my @WNIDs = @$idsRef;
#
#    if (scalar(@names) != scalar(@WNIDs)) { #Arry lengths are not equal
#        die "listParents: ERROR: Output arrays must be of equal length\n";
#    }
#
#    my $arryLength = scalar(@names);
#
#
#    #Set up temp dir
#    my $TMP_DIR = makeTempDir();
#
#    #print to screen
#    my $root = @names[-1];
#    for (my $i=0; $i<$arryLength; $i++) {
#        my $name = @names[$i];
#        my $wnid = @WNIDs[$i];
#        if ($name =~ m/$root/) {
#            print "$wnid\t\t$name\n";
#            print "\n\n----------\n";
#        } else {
#            print "$wnid\t$name\n";
#        }
#    }
#
#    #print to file
#    print "\nPrinting output to files...";
#    open (NameOut, ">", "$TMP_DIR/synsetParentNames.ssv") or die "Can't open file for writing! $TMP_DIR/synsetParentWNIDs.ssv\nError: $!"; #ssv = semi-colon separated values
#    open (WnidOut, ">", "$TMP_DIR/synsetParentWNIDs.ssv") or die "Can't open file for writing! $TMP_DIR/synsetParentNames.ssv\nError: $!"; #ssv = semi-colon separated values
#
#    my $firstRun = 1;
#    for (my $j=$arryLength-1; $j>=0; $j--) {
#        my $name = $names[$j];
#        my $wnid = $WNIDs[$j];
#
#        unless ($firstRun) {
#            if ($name =~ m/$root/) {
#                print NameOut "\n";
#                print WnidOut "\n";
#            }
#        }
#
#        print NameOut "$name";
#        print WnidOut "$wnid";
#
#        if ($j >= 1 ) { 
#            unless ($names[$j-1] =~ m/$root/) { #if we are not at the last item (next item will be root again)
#                print NameOut ";";
#                print WnidOut ";";
#            }
#        }
#        $firstRun = 0;
#    }
#
#    print "\nProgram Complete.\n";
#} else {
#    &lpPostUsage();
#}
#########################
#########################
 
sub lpPostUsage() {
    die "\n\nUsage:\n./listParents.pl input\nSupported inputs: synsetID, list.txt, list.html\n\n\n";
}

sub listParents ($) {
    use XML::XPath;
    use XML::XPath::XMLParser;

    use globalVars;
    my $useProxy  = getUseProxy globalVars();
    my $PROXY_URL = "";
    if ($useProxy) {
        $PROXY_URL = getProxyURL globalVars();
    }

    $USER_AGENT= "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)";
    $STRUCTURE_URL = "http://www.image-net.org/api/xml/structure_released.xml";

#Format input
    $userIn = $_[0];
    chomp($userIn);
    my $inputType = checkInputType($userIn);
    if ($inputType == 0) {
        &lpPostUsage();
    } elsif ($inputType == 3) {
        $userIn =~ s/\/$//g;
    }

#Set up temp dir
    my $TMP_DIR = makeTempDir();

#Download Image-Net structure if it does not already exist in the temp folder
    unless (-e "$TMP_DIR/structure.xml") {
        print "listParents: Downloading most current hierarchy from Image-Net...\n";
        if ($use_proxy) {
            system("curl -# -x \"$PROXY_URL\" \"$STRUCTURE_URL\" -A \"$USER_AGENT\" -o $TMP_DIR/structure.xml");
        } else {
            system("curl -# \"$STRUCTURE_URL\" -A \"$USER_AGENT\" -o $TMP_DIR/structure.xml");
        }
        print "listParents: Done.\n";
    }

    ######################################
    ## Uncomment below to print a file called structure.txt which has a better visualization of strcture.xml
    ######################################
    #use XML::Simple;
    #use Data::Dumper;
    #
    #$labels = XMLin("$TMP_DIR/structure.xml",KeyAttr => 'wnid');
    #print "listParents: Printing XML structure to text file...\n";
    #open(OUTFILE,">","$TMP_DIR/structure.txt");
    #print OUTFILE Dumper($labels);
    #close OUTFILE;
    #print "listParents: Done.\n";
    ######################################
    ######################################

    my @inArray;
    if ($inputType==1) { #The user has input a synset id
        @inArray = ($userIn);
    } elsif ($inputType==2) { #The user has input a text file full of synset ids
        #Open input file
        open(INTEXT,"<", $ARGV[0]) or die $!;
        @inArray = <INTEXT>;
        close(INTEXT);
    } elsif($inputType==3) { #The user has input a folder path
        print "listParents: Getting list of WNIDs from the input dir $userIn\n";
        my $ext = 'tar';
        print $userIn;
        my @fileList = findFiles($userIn,$ext);

        for (my $i=0; $i<scalar(@fileList); $i++) {
            if ($fileList[$i] =~ /(n\d+)/) {
                $fileList[$i] = $1;
            } else {
                delete $fileList[$i];
            }
        }

        my $numFilesFound = scalar(@fileList);
        print "listParents: Found $numFilesFound WNIDs.\n";

        @inArray = @fileList;
    } else {
        die "listParents: ERROR: inputType is not correct.\n";
    }

    my @parentTreeNames;
    my @parentTreeWNIDs;

#Parse through search terms, everything inside this loop is done for each search term
    my $count = 0;
    my $total = scalar(@inArray);
    foreach my $lineInput (@inArray) {

        chomp($lineInput);

        #Parse input line for synset ID
        if ($lineInput =~ m/(n\d+)/) {
            $input = $1;
        } else {
            next;
        }


        ######################################
        ##For now let's assume they only input synset IDs
        ######################################
        ###Decide if input was a WNID or a synset name
        #my ($wnid, $path) = 0;
        #if ($input =~ /^n[\d]+$/) { #WNID
        #    print "listParents: Finding the parents of WNID: \"$input\" in the Image-Net hierarchy.\n";
        #    $path = "//synset[\@wnid=\'${input}\']";
        #    $wnid = 1;
        #} else { #NAME
        #    print "listParents: Finding the parents of synset \"$input\" in the Image-Net hierarchy.\n";
        #    if ($input =~ /'/) {
        #        $path = "//synset[\@words=\"${input}\"]";
        #    } else {
        #        $path = "//synset[\@words=\'${input}\']";
        #    }
        #    $wnid = 0;
        #}
        ######################################
        $count += 1;
        print "\nlistParents: Finding the parents of WNID: \"$input\" in the Image-Net hierarchy ($count IDs out of $total).\n";
        my $path = "//synset[\@wnid=\'${input}\']";
        ######################################

        my $structure = "$TMP_DIR/structure.xml";
        my $xp = XML::XPath->new(filename => $structure);

        unless ($xp->exists($path)) {
            die "\n\nERROR: Couldn't find $input in $TMP_DIR/structure.xml!\n\tSearch Path: $path\n\n";
        }

        my $nodeset = $xp->find($path);

        my @nodelist = $nodeset->get_nodelist;

        foreach my $node (@nodelist) {
            push(@parentTreeNames,$node->getAttribute(words));
            push(@parentTreeWNIDs,$node->getAttribute(wnid));

            my $parent = $node->getParentNode();

            #Grab all parents & grandparents in order. path var ancestor-or-self::node() will not return them in order.
            while ($parent and $parent->getAttribute(words)) {
                push(@parentTreeNames,$parent->getAttribute(words));
                push(@parentTreeWNIDs,$parent->getAttribute(wnid));
                $parent = $parent->getParentNode();
            }
        }

        print "\n";
    }
    return (\@parentTreeNames,\@parentTreeWNIDs);
}
1;
