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
##
############


#####
##Uncomment _below_ to run from command line
##Leave _below_ commented in order to call this function from another program
#if ($ARGV[0]) {
#    @OUTPUT = &listChildren($ARGV[0]);
#    #print "\n\nlistChildren: OUTPUT:\n\n",
#    #      join("\n",@OUTPUT),"\n\n\n";
#
#    $input = @OUTPUT[0];
#    foreach $item (@OUTPUT) {
#        if ($item =~ m/$input/) {
#            print "\n\n----------\n";
#        }
#        print "$item\n";
#    }
#    print "\n";
#} else {
#    die "Usage: ./listChildren.pl \"category\"\n";
#}
#####
 
sub listChildren ($) {
    use XML::XPath;
    use XML::XPath::XMLParser;

    $USER_AGENT= "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)";
    $STRUCTURE_URL = "http://www.image-net.org/api/xml/structure_released.xml";

#Format input
    if ($_[0]) {
        $input = $_[0];
        chomp($input);
    } else {
        die "Usage: ./listChildren.pl synsetID\n";
    }

#Set up output dir
    my $currDir = `pwd`;
    chomp($currDir);
    $currDir =~ s/\s/\\ /g;
    my $TMP_DIR = "$currDir/temp";
    unless (-d $TMP_DIR) {
        system("mkdir -p $TMP_DIR") or die "listChildren: Couldn't make dir $TMP_DIR!\n";
    }

#Download Image-Net structure if it does not already exist in the temp folder
    unless (-e "$TMP_DIR/structure.xml") {
        print "listChildren: Downloading most current hierarchy from Image-Net...\n";
        system("curl -# \"$STRUCTURE_URL\" -A \"$USER_AGENT\" -o $TMP_DIR/structure.xml");
        print "listChildren: Done.\n";
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
##        print "listChildren: Finding the children of WNID: \"$input\" in the Image-Net hierarchy...\n";
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
    print "listChildren: Finding the children of WNID: \"$input\" in the Image-Net hierarchy...";
    my $path = "//synset[\@wnid=\'${input}\']/descendant-or-self::node()";
######################################

    my $structure = "$TMP_DIR/structure.xml";
    my $xp = XML::XPath->new(filename => $structure);

    unless ($xp->exists($path)) {
        die "\n\nERROR: Couldn't find $input in $TMP_DIR/structure.xml!\n\tSearch Path: $path\n\n";
    }

    my $nodeset = $xp->find($path);

    #print "\nParent tree(s):\n-------------------\n";

    my @nodelist = $nodeset->get_nodelist;

    my @childTreeNames;
    my @childTreeWNIDs;

    #$selfNode = @nodelist[0]->getAttribute(wnid);

    my %seen = ();

    foreach my $node (@nodelist) {

        next if $seen{$node->getAttribute(wnid)}++;
        push(@childTreeNames,$node->getAttribute(words));
        push(@childTreeWNIDs,$node->getAttribute(wnid));

        #if ($node->getAttribute(wnid) =~ m/$selfNode/) {
        #    print "-------------------\n";
        #}
        #print "\t",$node->getAttribute(words);
        #print "\t",$node->getAttribute(WNID),"\n";
    }

    print "Done.\n";
    return (\@childTreeNames,\@childTreeWNIDs);
}
1;
