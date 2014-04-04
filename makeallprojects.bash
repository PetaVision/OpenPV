#! /bin/bash
# A bash script to enter each directory in the directory containing this
# script, in turn; selecting those that contain a Makefile, and run
# (make clean && make) for those that do.
# Finally, it reports which directories don't have a Makefile and which
# directories had the make command fail.

# Navigate to directory containing systems tests.
scriptdir=$(dirname "$0")

fails=""
dne=""
while read proj # will use a here-string at the end of the while statement for input
do
    baseproj="$(basename $proj)"
    if test -f "$proj/Makefile"
    then
        if ! (make --directory="$proj" clean && make --directory="$proj")
        then
            fails="$fails $baseproj"
        fi
    else
        dne="$dne $baseproj"
    fi
done <<< "$(find "$scriptdir" -d 1 -type d \! -path "$scriptdir/CMakeFiles")"

status=0
if test -n "$fails"
then
    echo "The following projects failed to make: $fails"
    status=1
else
    echo "All projects built."
fi

if test -n "$dne"
then
    echo "The following projects do not have a Makefile: $dne"
    status=1
fi

if test status != 0
then
    exit 1
fi
