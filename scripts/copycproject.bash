#! /bin/bash
# Copies either the mac.cproject or linux.cproject file of each project

# The script should be in PetaVision/scripts relative to the eclipse working directory (ewd)
if test "${0%/*}" != "$0"
then
    ewd="${0%/*}"
else
    ewd="."
fi
ewd=$ewd/../.. # $ewd should point to the eclipse workspace directory

# Get machine type, either mac or linux.  uname -s returns Darwin or Linux.
machtype="$(uname -s | tr [:upper:] [:lower:])"
case $machtype in
    (darwin) cprojecttype=mac;;
    (linux)  cprojecttype=linux;;
esac

for k in $(ls $ewd) # this assumes none of the project names have spaces in them
do
    cp -p $ewd/$k/$cprojecttype.cproject $ewd/$k/.cproject
done