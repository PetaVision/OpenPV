#!/bin/bash

path=$1 #Should be absolute
year=$2
month=$3
day=$4

videoList=$(ls -1d $path/$year/$month/$day/*/*)
#echo "$videoList"

sedCommand='s/.*, \(.*\) fps.*/\1/p'
outPath="$path/frames/${year}_${month}_${day}"
mkdir -p "$outPath"

for k in $videoList
do
    r=$(ffmpeg -i $k 2>&1 | sed -n "s/.*, \(.*\) fps.*/\1/p")
    baseName=$(basename "$k" .mp4)
    echo "$baseName"
    ffmpeg -i "$k" -r "$r" "${outPath}/${baseName}-%04d.png"
done

