#!/usr/bin/env perl

############
## listChildren.pl
##
## Written by:
##      Dylan Paiton
##      Los Alamos National Laboratory, Group ISR-2 
##      paiton@lanl.gov
##
## Download Image-Net structure
## Locate $input in structure
## Return all children of $input, along all trees
##
## TODO:
##   Have flag for each output type. One for downloadImages.pl and one for collectTargetImages.pl
##
############

#####
##Uncomment _below_ to run from command line
##Leave _below_ commented in order to call this function from another program
#####
#if ($ARGV[0]) {
#    my ($namesRef, $idsRef) = &listChildren($ARGV[0]);
#
#    my @names = @$namesRef;
#    my @WNIDs = @$idsRef;
#
#    if (scalar(@names) != scalar(@WNIDs)) { #Arry lengths are not equal
#        die "listChildren: ERROR: Output arrays must be of equal length\n";
#    }
#
#    my $arrayLength = scalar(@names);
#    my $rootName   = @names[0];
#    my $rootWNID   = @WNIDs[0];
#
#    #Set up temp dir
#    my $TMP_DIR = makeTempDir();
#
#    #print to screen
#    for (my $i=$arrayLength-1; $i>0; $i--) {
#        my $name = @names[$i];
#        my $wnid = @WNIDs[$i];
#        if ($name =~ m/$rootName/) { #If we're at the root
#            print "\n\n-------------------------------------------------\n";
#        }
#        print "$wnid\t$name\n";
#    }
#
#    #print to file
#    print "\nPrinting output to files...";
#    open (NameOut, ">", "$TMP_DIR/${rootWNID}_Children_Names.ssv") or die "listChildren: Can't open file for writing! $TMP_DIR/${rootWNID}_Children_Names.ssv\nError: $!"; 
#    open (WnidOut, ">", "$TMP_DIR/${rootWNID}_Children_WNIDs.ssv") or die "listChildren: Can't open file for writing! $TMP_DIR/${rootWNID}_Children_WNIDs.ssv\nError: $!"; 
#
#    my $firstRun = 1;
#    for (my $j=$arrayLength-1; $j>0; $j--) {
#        my $name = $names[$j];
#        my $wnid = $WNIDs[$j];
#
#        unless ($firstRun) {
#            if ($wnid =~ m/$rootWNID/) {
#                print NameOut "\n";
#                print WnidOut "\n";
#            }
#        }
#
#        print NameOut  "$name";
#        print WnidOut  "$wnid";
#
#        if ($j >= 1) {
#            unless ($WNIDs[$j-1] =~ m/$rootWNID/) { #if we are not at the last item (next item will be root again)
#                print NameOut ";";
#                print WnidOut ";";
#            }
#        }
#        $firstRun = 0;
#    }
#
#    print "\nProgram Complete.\n";
#} else {
#    &lcPostUsage();
#}
#####
 
sub lcPostUsage() {
    die "\n\nUsage: ./listChildren.pl synsetID\n\n\n";
}

sub listChildren ($) {
    use XML::XPath;
    use XML::XPath::XMLParser;

    use globalVars;
    my $useProxy = getUseProxy globalVars();
    my $PROXY_URL = "";
    if ($useProxy) {
        $PROXY_URL = getProxyURL globalVars();
    }

    require 'makeTempDir.pl';

    $USER_AGENT= "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)";
    $STRUCTURE_URL = "http://www.image-net.org/api/xml/structure_released.xml";

#Format input
    if ($_[0]) {
        $input = $_[0];
        chomp($input);
    } else {
        &lcPostUsage();
    }

#Set up temp dir
    my $TMP_DIR = makeTempDir();

#Download Image-Net structure if it does not already exist in the temp folder
    unless (-e "$TMP_DIR/structure.xml") {
        print "listChildren: Downloading most current hierarchy from Image-Net...\n";
        if ($use_proxy) {
            system("curl -# -x \"$PROXY_URL\" \"$STRUCTURE_URL\" -A \"$USER_AGENT\" -o $TMP_DIR/structure.xml");
        } else {
            system("curl -# \"$STRUCTURE_URL\" -A \"$USER_AGENT\" -o $TMP_DIR/structure.xml");
        }
    }

######################################
## Uncomment below to print a file called structure.txt which has a better visualization of strcture.xml
######################################
##  use XML::Simple;
##  use Data::Dumper;
##
##  $labels = XMLin("$TMP_DIR/structure.xml",
##          KeyAttr => 'wnid');
##  print "listChildren: Printing XML structure to text file...\n";
##  open(OUTFILE,">","$TMP_DIR/structure.txt");
##  print OUTFILE Dumper($labels);
##  close OUTFILE;
##  print "listChildren: Done.\n";
######################################
######################################

######################################
##For now let's assume they only input synset IDs
######################################
###Decide if input was a WNID or a synset name
##    my ($wnid, $path) = 0;
##    if ($input =~ /^n[\d]+$/) { #WNID
##        print "listChildren: Finding the children of WNID \"$input\" in the Image-Net hierarchy...\n";
##        $path = "//synset[\@wnid=\'${input}\']";
##        $wnid = 1;
##    } else { #NAME
##        print "listChildren: Finding the children of synset \"$input\" in the Image-Net hierarchy...\n";
##        if ($input =~ /'/) {
##            $path = "//synset[\@words=\"${input}\"]";
##        } else {
##            $path = "//synset[\@words=\'${input}\']";
##        }
##        $wnid = 0;
##    }
######################################
    print "listChildren: Finding the children of WNID \"$input\" in the Image-Net hierarchy.\n";
    my $path = "//synset[\@wnid=\'${input}\']/descendant-or-self::node()";
######################################

    my $structure = "$TMP_DIR/structure.xml";
    my $xp = XML::XPath->new(filename => $structure);

    unless ($xp->exists($path)) {
        die "\n\nERROR: Couldn't find $input in $TMP_DIR/structure.xml!\n\tSearch Path: $path\n\n";
    }

    my $nodeSet  = $xp->find($path);
    my @nodeList = $nodeSet->get_nodelist;

    my $rootName = @nodeList[0]->getAttribute(words);
    my $rootWNID = @nodeList[0]->getAttribute(wnid);

    my @childTreeNames;
    my @childTreeWNIDs;

    my %seen = ();
    $seen{$rootWNID}++;

    foreach my $node (@nodeList) {
        next if $seen{$node->getAttribute(wnid)}++;

        #push child
        push(@childTreeNames,$node->getAttribute(words));
        push(@childTreeWNIDs,$node->getAttribute(wnid));

        #push all children 
        my $parent = $node->getParentNode();
        while (lc($parent->getAttribute(wnid)) ne lc($rootWNID) and $parent->getAttribute(wnid)) {
            push(@childTreeNames,$parent->getAttribute(words));
            push(@childTreeWNIDs,$parent->getAttribute(wnid));
            $parent = $parent->getParentNode();
        }

        #push root
        push(@childTreeNames,$rootName);
        push(@childTreeWNIDs,$rootWNID);
    }

    return (\@childTreeNames,\@childTreeWNIDs);
}
1;
