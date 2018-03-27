#!/usr/bin/env sh
case "$1" in
    --help | -h | --usage | -u)
        echo "$0 /path/to/timing/file.txt"
        echo "sorts the timer information in a file by elapsed time"
        echo "The input file can be a timers.txt file from a checkpoint"
        echo "or the log file of a PetaVision run."
        exit 0;;
    *)
        egrep "processor cycle time" "$1" |
        sed -e '1,$s/://g' |
        sort -g -k 11;;
esac
