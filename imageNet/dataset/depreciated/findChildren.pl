#!/usr/bin/env perl

############
## findChildren.pl
##
## Written by:
##      Dylan Paiton
##      Los Alamos National Laboratory, Group ISR-3 
##      paiton@lanl.gov
##
## Similar to findParents.pl
## Download Image-Net structure
## Locate $input in structure
## Return all children of $input
##
## TODO:
##
############


##Uncomment below to run from command line
##This must stay commented in order to call this function from another program
if ($ARGV[0]) {
    @OUTPUT = &findChildren($ARGV[0]);
    print "\n\nOUTPUT:\n",
          join("\n",@OUTPUT),
          "\n";
} else {
    die "Usage: ./findChildren.pl \"category\"\n";
}

sub findChildren{
    use XML::XPath;
    use XML::XPath::XMLParser;

    $USER_AGENT= "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)";
    $STRUCTURE_URL = "http://www.image-net.org/api/xml/structure_released.xml";

    $currDir = `pwd`;
    chomp($currDir);
    $currDir =~ s/\s/\\ /g;
    $TMP_DIR = "$currDir/../tmp";

    unless (-d $TMP_DIR) {
        system("mkdir -p $TMP_DIR");
    }

    undef @CHILDREN;

    $input = $_[0];
    print "\nFinding \"$input\" in the Image-Net hierarchy...\n";

#Download Image-Net structure if it does not already exist in the temp folder
    unless (-e "$TMP_DIR/structure.xml") {
        print "Downloading most current word labels...\n";
        system("curl -# \"$STRUCTURE_URL\" -A \"$USER_AGENT\" -o $TMP_DIR/structure.xml");
        print "Done.\n";
    }

    $file = "$TMP_DIR/structure.xml";
    $xp = XML::XPath->new(filename => $file);


#Find input node, and all children using descendant-or-self
    if ($path =~ /'/) {
        $path = "//synset[\@words=\"${input}\"]/descendant-or-self::node()";
    } else {
        $path = "//synset[\@words=\'${input}\']/descendant-or-self::node()";
    }

    if ($xp->exists($path)) {
        print "The synset \"$input\" exists in the XML file. Searching for children...\n";
        
        $nodeset = $xp->find($path);

        my %seen = ();
        foreach $node ($nodeset->get_nodelist) {
            next if $seen{ $node->getAttribute(words) }++;
            push(@CHILDREN,$node->getAttribute(words));
        }
    } else {
        die "\n\nERROR: $input doesn't exist in $TMP_DIR/structure.xml!\n\n";
    }

    print "Done. Returning child nodes.\n";
    return @CHILDREN;
}
1;
