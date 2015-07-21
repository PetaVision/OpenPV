#!/usr/bin/env perl

############
## findParents.pl
##
## Written by:
##      Dylan Paiton
##      Los Alamos National Laboratory, Group ISR-2 
##      paiton@lanl.gov
##
## Similar to findChildren.pl
## Download Image-Net structure
## Locate $input in structure
##  Many inputs exist in multiple hierarchies, user must choose which hierarchy to follow
## Return all parents of $input, within specified hierarchy
##
## TODO:
##
############


##Uncomment below to run from command line
##This must stay commented in order to call this function from another program
if ($ARGV[0]) {
    @OUTPUT = &findParents($ARGV[0]);
    print "\n\nOUTPUT:\n",
          join("\n",@OUTPUT),
          "\n\n\n";
} else {
    die "Usage: ./findParents.pl \"category\"\n";
}
 
sub findParents {
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

    undef @RETURN;
    undef @PARENTS;

    if ($_[0]) {
        $input = $_[0];
    } else {
        die "\n\nERROR: Invalid input to &findParents\nInput: \"$_[0]\"\n\n";
    }
    if ($_[1]) {
        @OLDPARENTS = @_;
        my $sh = shift(@OLDPARENTS);
        $sh = shift(@OLDPARENTS);
        @OLDPARENTS = reverse(@OLDPARENTS);
    }

#Decide if input was a WNID or a synset name
    chomp($input);
    my ($wnid, $path) = 0;
    if ($input =~ /^n[\d]+$/) { #WNID
        print "\nFinding the parents of WNID: \"$input\" in the Image-Net hierarchy...\n";
        $path = "//synset[\@wnid=\'${input}\']";
        $wnid = 1;
    } else { #NAME
        print "\nFinding the parents of synset \"$input\" in the Image-Net hierarchy...\n";
        if ($input =~ /'/) {
            $path = "//synset[\@words=\"${input}\"]";
        } else {
            $path = "//synset[\@words=\'${input}\']";
        }
        $wnid = 0;
    }

#Download Image-Net structure if it does not already exist in the temp folder
    unless (-e "$TMP_DIR/structure.xml") {
        print "\nDownloading most current word labels...\n";
        system("curl -# \"$STRUCTURE_URL\" -A \"$USER_AGENT\" -o $TMP_DIR/structure.xml");
        print "Done.\n";
    }

######################################
##
## Uncomment below to print a file called structure.txt which has a better visualization of strcture.xml
##
##  use XML::Simple;
##  use Data::Dumper;
##
##  $labels = XMLin("$TMP_DIR/structure.xml",
##          KeyAttr => 'wnid');
##  print "Printing XML structure to text file...\n";
##  open(OUTFILE,">","$TMP_DIR/structure.txt");
##  print OUTFILE Dumper($labels);
##  close OUTFILE;
##  print "Done.\n";
##
######################################
    
    $structure = "$TMP_DIR/structure.xml";
    $xp = XML::XPath->new(filename => $structure);

    my ($i, $pathExists, $secondtry) = 0;
    my $firsttry = 1;
    my $searchterm = $input;
    
#Search for path in XML file, if not found then modify input and try again
    while ($pathExists == 0) {
        if ($xp->exists($path)) {
            print "\"$searchterm\" exists in the XML file.\n";
            $searchterm = $input;
            $pathExists = 1;
        } else {
            if ($wnid) {
                die "\n\nERROR: Couldn't find $input in $TMP_DIR/structure.xml!\n\n";
            } else {
                if ($firsttry) {
                    print "\nWARNING: Couldnt find \"$input\" in $TMP_DIR/structure.xml\n",
                          "Searching to see if \"$input\" is part of a larger name...\n";
                    $searchterm = $input;
                    $firsttry = 0;
                    $secondtry = 1;
                } elsif ($secondtry) {
                    $searchterm = lcfirst($input);
                    $secondtry = 0;
                } else {
                    print "\nWARNING: $searchterm wasn't found.\n";
                    @INPUTS = split(", ",$searchterm);
                    @RINPUTS = reverse(@INPUTS);
                    if ($i < scalar(@RINPUTS)) {
                        $searchterm =~ s/, $RINPUTS[$i]//g;
                        print "Searching for $searchterm instead.\n\n";
                        $i += 1;
                    } else {
                        die "\n\nERROR: Couldn't find $input in $TMP_DIR/structure.xml!\n\n";
                    } #$i < scalar(@RINPUTS)
                } #firsttry
            } #wnid
            $path = "//synset[contains(\@words,'${searchterm}')]";
        } #xp->exists(path)
    } #while

#If the path exists, find the nodes associated with it, list the parents, and add them to an array
    $nodeset = $xp->find($path);
    print "Parent tree(s):\n-------------------\n";
    $line = 0;
    $option = 1;
    @NODELIST = $nodeset->get_nodelist;
    push(@LINEINFO, "0");
#Look through all nodes at $path
    foreach $node (@NODELIST) {
        if ($node->getAttribute(words) =~ /$input/) {

            print "$option\n";
            print "\t",$node->getAttribute(words),"\n";

            push(@INITIALR,$node->getAttribute(words));
            push(@INITIALH,$node);

#Grab all parents & grandparents. There is an easier way to do this using ancestor-or-self::, but I didn't know that when I wrote this.
            if ($parent = $node->getParentNode()) {
                unless ($line) {
                    $line += 1;
                }
            }
            while ($parent and $parent->getAttribute(words)) {
                print "\t",$parent->getAttribute(words),"\n";
                push(@TREELIST,$parent->getAttribute(words));
                push(@FULLHIERARCHY,$parent);
                if ($parent = $parent->getParentNode()) {
                    $line += 1;
                }
            }
            push (@LINEINFO, $line-1);
            $option += 1;
            print "-------------------\n";
        } else {
           #print "WARNING: Didn't find $input in \n",
           #      XML::XPath::XMLParser::as_string($node),
           #      "\n\n";
        }
    }
    undef(@NODELIST);

#If user input included a list of the previous parents used, try to match current options with previous parents
    my ($optionChoice, $askuser) = 1;
    if (@OLDPARENTS) {
        print "The term exists in more than one tree. Attempting to match parents with previous term...\n";
        for (my $i=1; $i<$option; $i++) {
            for (my $j=1; $j<=scalar(@LINEINFO); $j++) {
                if ($j == $i) {
                    for ($k=$LINEINFO[$j-1]; $k<$LINEINFO[$j]; $k++) {
                        push(@NEWTREE,$TREELIST[$k]);
                    }
                }
            }

            my $idx = scalar(@NEWTREE) - 1;
            my $sameparent = 0;

            foreach my $item (@OLDPARENTS) {
                if ($item =~ /^$NEWTREE[$idx]$/) {
                    $sameparent = 1;
                    if ($idx >= 0) {
                        $idx = $idx - 1;
                    } else {
                       last; 
                    }
                    next;
                } else {
                    $sameparent = 0;
                    last;
                }
            }#foreach $item
            if ($sameparent) {
                push(@optionChoices, $i);
                $askuser = 0;
            } else {
                $askuser = 1;
            }
            undef(@NEWTREE);
        }#for $i
        if ($askuser) {
            print "Failed to match parents.\n";
        }
    } else {
        $askuser = 1;
    }

#If unable to find previous parents, or if previous parents were not given, ask user which parental tree to use
    if ($askuser) {
        if ($option-1 == 1) {
            $optionChoice = 1;
        } else {
            print "Please select a hierarchy [1]: ";
            $optionChoice = <STDIN>;
            chomp($optionChoice);
            unless (($optionChoice > 0) && ($optionChoice <= $option-1)) {
                print "Using default value.\n";
                $optionChoice = 1;
            }
        }
    } else {
        if (scalar(@optionChoices) == 1) {
            $optionChoice = $optionChoices[0];
            print "Option $optionChoice was chosen because it has the same parents as the previous input.\n";
        } else {
            print "There are multiple trees which have the same parents as your previous tree.\n",
                  "Please select one of the following options: ",
                  join(" ",@optionChoices),
                  " [$optionChoices[0]] ";

            $optionChoice = <STDIN>;
            chomp($optionChoice);
            if (grep $_ eq $optionChoice, @optionChoices) {
                print "Using option $optionChoice\n";
            } else {
                print "Invalid input, using default value.\n";
                $optionChoice = $optionChoices[0];
                print "Option $optionChoice was chosen because it has the same parents as the previous input.\n";
            }
        }
    }

    push(@RETURN,@INITIALR[$optionChoice-1]);
    push(@HIERARCHY,$INITIALH[$optionChoice-1]);

    undef @OLDPARENTS;
    undef @optionChoices;
    undef @INITIALR;
    undef @INITIALH;

#Push chosen parental tree to arrays for output
    for (my $j=1; $j<=scalar(@LINEINFO); $j++) {
        if ($j == $optionChoice) {
            for ($k=$LINEINFO[$j-1]; $k<$LINEINFO[$j]; $k++) {
                push(@HIERARCHY,$FULLHIERARCHY[$k]);
                push(@RETURN,$TREELIST[$k]);
            }
        }
    }

    undef @TREELIST;
    undef @FULLHIERARCHY;
    undef @LINEINFO;

#Write out new XML file to contain the chosen hierarchy
    @RHIERARCHY = reverse(@HIERARCHY);
    undef @HIERARCHY;
    open(OUTFILE,">","$TMP_DIR/temp.xml");
    if (-e "$TMP_DIR/folderStructure.xml") {
        print "Parents found. Writing to $TMP_DIR/folderStructure.xml\n";
        open(INFILE,"<","$TMP_DIR/folderStructure.xml");
        @INFILE = <INFILE>;
        close(INFILE);

        ($nodenum, $lnum, $exists, $nodepos) = 0;
        foreach $line (@INFILE) {
            if ($nodenum < scalar(@RHIERARCHY)-1) {
                $attrib = $RHIERARCHY[$nodenum]->getAttribute('wnid');
                if ($line =~ /$attrib/) {
                    $exists = 1;
                    $nodenum += 1;
                    $nodepos = $lnum;
                }
            } elsif ($nodenum == scalar(@RHIERARCHY)-1) {
                $attrib = $RHIERARCHY[$nodenum]->getAttribute('wnid');
                if ($line =~ /$attrib/) {
                    $exists = 2;
                    last;
                } 
            } else {
                $lnum += 1;
                next;
            }
            $lnum += 1;
        }
        if ($exists == 1) {
            $lpos = 0;
            foreach $line(@INFILE){
                if ($lpos <= $nodepos) {
                    $lpos += 1;
                    print OUTFILE $line;
                } else {
                    last;
                }
            }
            $npos = 0;
            $numprints=0;
            foreach $node(@RHIERARCHY) {
                if ($npos >= $nodenum) {
                    $rootName = $node->getName();
                    push(@ROOTNAMES,$rootName);
                    @rootAttribs = $node->getAttributes();

                    print OUTFILE "<$rootName";
                    foreach $attrib (@rootAttribs) {
                        $aname = $attrib->getName();
                        $aval = $attrib->getNodeValue();
                        $aval =~ s/\&/\&amp\;/g;
                        $aval =~ s/\"/\&quot\;/g;
                        $aval =~ s/\'/\&apos\;/g;
                        $aval =~ s/\</\&lt\;/g;
                        $aval =~ s/\>/\&gt\;/g;
                        print OUTFILE " $aname=\"$aval\"";
                    }
                    print OUTFILE ">\n";
                    $numprints += 1;
                } 
                $npos += 1;
            }
            for($i=0;$i<$numprints;$i++) {
                print OUTFILE "</synset>\n";
            }
            for($i=$lpos; $i<=scalar(@INFILE); $i++) {
                print OUTFILE $INFILE[$i];
            }
        } elsif ($exists == 2) {
            print "The directory already exists in the XML file.\n";
            foreach $line (@INFILE) {
                print OUTFILE $line;
            }
        } else {
            foreach $line (@INFILE) {
                if ($line =~ /\/NewStructure/) {
                    next;
                }
                print OUTFILE $line;
            }
            foreach $node(@RHIERARCHY) {
                $rootName = $node->getName();
                push(@ROOTNAMES,$rootName);
                @rootAttribs = $node->getAttributes();

                print OUTFILE "<$rootName";
                foreach $attrib (@rootAttribs) {
                    $aname = $attrib->getName();
                    $aval = $attrib->getNodeValue();
                    $aval =~ s/\&/\&amp\;/g;
                    $aval =~ s/\"/\&quot\;/g;
                    $aval =~ s/\'/\&apos\;/g;
                    $aval =~ s/\</\&lt\;/g;
                    $aval =~ s/\>/\&gt\;/g;
                    print OUTFILE " $aname=\"$aval\"";
                }
                print OUTFILE ">\n";
            }
            foreach $rootName (@ROOTNAMES) {
                print OUTFILE "</$rootName>\n";
            }
            print OUTFILE "</NewStructure>\n";
        }
    } else {
        print "Parents found. Writing to a new XML file...\n";
        print OUTFILE "<NewStructure>\n";
        foreach $node (@RHIERARCHY) {
            $rootName = $node->getName();
            push(@ROOTNAMES,$rootName);
            @rootAttribs = $node->getAttributes();
            print OUTFILE "<$rootName";
            foreach $attrib (@rootAttribs) {
                $aname = $attrib->getName();
                $aval = $attrib->getNodeValue();
                $aval =~ s/\&/\&amp\;/g;
                $aval =~ s/\"/\&quot\;/g;
                $aval =~ s/\'/\&apos\;/g;
                $aval =~ s/\</\&lt\;/g;
                $aval =~ s/\>/\&gt\;/g;
                print OUTFILE " $aname=\"$aval\"";
            }
            print OUTFILE ">\n";
        }
        foreach $rootName (@ROOTNAMES) {
            print OUTFILE "</$rootName>\n";
        }
        print OUTFILE "</NewStructure>\n";
    }
    close OUTFILE;
    undef @INFILE;
    undef @ROOTNAMES;
    undef @rootAttribs;
    system("mv -f $TMP_DIR/temp.xml $TMP_DIR/folderStructure.xml");

    print "Done. Returning parents.\n";

    return @RETURN;
}
1;
