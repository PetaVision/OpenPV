#!/usr/bin/env perl

############
## getPath.pl
##
## Written by:
##      Dylan Paiton
##      Los Alamos National Laboratory, Group ISR-2 
##      paiton@lanl.gov
##
## Uses XML script created by findParents.pl to find all of the parents
##  and output them as a string which represents the file path to the given input.
##
## TODO:
##
############

##Uncomment below to run from command line
##This must stay commented in order to call this function from another program
#if ($ARGV[0] && $ARGV[1]) {
#    $output = &getPath($ARGV[0], $ARGV[1]);
#    print "\n\nOUTPUT:\n",
#          $output,
#          "\n\n\n";
#} else {
#    print "WARNING: &getPath should get exactly two arguments!\n";
#    die "Usage: ./getPath.pl \"dest_dir\" \"category\"\n";
#}

sub getPath{
    use XML::XPath;
    use XML::XPath::XMLParser;

    my $currDir = `pwd`;
    chomp($currDir);
    $currDir =~ s/\s/\\ /g;
    my $TMP_DIR = "$currDir/../tmp";

    unless (-d $TMP_DIR) {
        system("mkdir -p $TMP_DIR");
    }

    my $IMG_DIR = $_[0];
    my $input = $_[1];

    print "\nFinding the path to \"$input\" in the $IMG_DIR/ folder...\n";

    unless (-d $TMP_DIR) {
        system("mkdir -p $TMP_DIR");
    }

    unless (-e "$IMG_DIR/folderStructure.xml") {
        die "folderStructure.xml does not exist. Can't find children.";
    }

    $file = "$IMG_DIR/folderStructure.xml";
    print "Parsing $file\n";

    $xp = XML::XPath->new(filename => $file);

    my $path = '';

    if ($input =~ /'/) {
        $path = "//synset[\@words,"${input}"]";
    } else {
        $path = "//synset[\@words,'${input}']";
    }

    $nodeset = $xp->find($path);
    $initial = 1;
    foreach $node ($nodeset->get_nodelist) {
        if (XML::XPath::XMLParser::as_string($node) =~ /($input)/) {
            if ($initial) {
                push(@DIRS,$node->getAttribute(words));
                $initial = 0;
            }
            $parent = $node->getParentNode();
            while ($parent and $parent->getParentNode() and $parent->getAttribute(words)) {
                push(@DIRS,$parent->getAttribute(words));
                $parent = $parent->getParentNode();
            }
        }
    }
    @RDIRS = reverse(@DIRS);
    undef @DIRS;
    $dir = $IMG_DIR;
    foreach $item(@RDIRS) {
        $dir .= "/" . $item;
    }

    print "Done. Returning directory.\n";
    
    return $dir;
}
1;
