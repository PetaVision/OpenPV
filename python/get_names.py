# Written by Max Theiler, max.theiler@gmail.com
# 2/5/2016

# An ancilliary script for analysis_parse.py and
# analyze_network.m, very likely to be obsolete.
# Should be deleted when those two commands are
# no longer being used.

from os import listdir
import re
import sys

llines = listdir(sys.argv[1] + "/src/layers")
clines = listdir(sys.argv[1] + "/src/connections")
layers = []
conns = []

for i in llines:
    j = re.search(".+(?=cpp)", i)
    if j:
        k = re.search("\w+", j.group())
        if k:
            layers.append(k.group())
for i in clines:
    j = re.search(".+(?=cpp)", i)
    if j:
        k = re.search("\w+", j.group())
        if k:
            conns.append(k.group())



print(layers)
print(conns)

f = open("layers.txt", "w")
for i in layers:
    f.write(i + "\n")
f.close()
f = open("connections.txt", "w")
for i in conns:
    f.write(i + "\n")
f.close()
