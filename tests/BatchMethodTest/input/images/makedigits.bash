#! /usr/bin/env bash
for k in {0..9}
do
    /usr/bin/convert -depth 8 -background black -fill white \
        -size 32x32 -pointsize 32 -gravity center label:$k $k.png
done
