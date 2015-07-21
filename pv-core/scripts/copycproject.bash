#! /bin/bash
# Usage: "bash copycproject.bash eclipse" or "bash copycproject.bash cmake"
# Copies a *.cproject file of each project
# The file name copied has the form <platform>-<makefilemanager>.cproject
# <platform> is either mac or linux, depending on the platform the script is run on
# <makefilemanager> is either eclipse or cmake; it is specified as the argument

# Determine makefilemanager
case ${1:-NONE} in
    (eclipse) makefilemanager="eclipse"
              ;;
    (cmake)   makefilemanager="cmake"
              ;;
    (NONE)    makefilemanager="eclipse";
              echo "Using copycproject with no arguments is deprecated."
              echo "Give the argument \"eclipse\" or \"cmake\" to specify the makefile manager." >&2;
              echo "Using argument \"eclipse\"."
              ;;
    (*)       echo "Usage: "bash copycproject.bash eclipse" or "bash copycproject.bash cmake"" >&2;
              exit 1
              ;;
esac

# Get machine type, either mac or linux.  uname -s returns Darwin or Linux.
machtype="$(uname -s | tr [:upper:] [:lower:])"
case $machtype in
    (darwin) cprojecttype=mac;;
    (linux)  cprojecttype=linux;;
    (*) echo "Unrecognized machine type \"$machtype\"" >&2; 
        exit 1;;
esac

# The script should be in PetaVision/scripts relative to the eclipse working directory (ewd)
if test "${0%/*}" != "$0"
then
    ewd="${0%/*}"
else
    ewd="."
fi
ewd=$ewd/../.. # $ewd should point to the eclipse workspace directory

for k in $(ls $ewd) # this assumes none of the project names have spaces in them
do
    if test -f $ewd/$k/$cprojecttype-$makefilemanager.cproject
    then
        cp -p $ewd/$k/$cprojecttype-$makefilemanager.cproject $ewd/$k/.cproject
    else
        echo "Warning: $ewd/$k/$cprojecttype-$makefilemanager.cproject does not exist" >&2;
    fi
done
