#!/bin/bash

path=$1 #Should be absolute
year=$2
month=$3
day=$4

a=$(ls -1d $path/$year/$month/$day/*/*)

echo $a
