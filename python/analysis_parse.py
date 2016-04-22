# Written by Max Theiler, max.theiler@gmail.com
# 2/5/2016

# This script is an obsolete, earlier version of the
# param_draw.py script that is used for the param-
# reading aspects of the analyze_network.m script in
# util/mlab. If no one is still using analyze_network.m,
# it should be deleted.



from os import listdir
import os
import re
import sys
import math

############################################################
######################## INPUT #############################
############################################################

# Input files and sources for connection/layer name:

layer_loc = ("./layers.txt")
conn_loc = ("./connections.txt")
output_dir = (".")

# Find a .params file, and throw an error if more than one is found.
foundparams = []
param_location = ""
print(listdir(output_dir))
for i in listdir(output_dir):
    if re.search("^\.",i):
        continue
    if re.search(".+\.params$",i):
        foundparams.append(re.search(".+\.params$",i).group())
if len(foundparams) > 1:
    print("Warning: Multiple params files found. Parsing " + foundparams[0])
    param_location = "./" + foundparams[0]
elif len(foundparams) == 0:
    print("Warning: No params files found in output directory.")
else:
    param_location = "./" + foundparams[0]
    print("Parsing " + foundparams[0])

# Lists of variables to look for in each layer and conn, plus defaults if they are not found.
print(param_location)
print(param_location)
print(param_location)
print(param_location)
print(param_location)

layer_vars = ["originalLayerName","phase"]
conn_vars = ["channelCode","maskLayerName","originalConnName","delay","plasticityFlag"]
layer_default = ["",""]
conn_default = ["","","","",""]
layer_ignore = ["",""]
conn_ignore = ["","","","0.000000",""]




############################################################
####################### PARSING ############################
############################################################

# Hardwire pre and post as the starting elements of the Conn lists

conn_vars = ["preLayerName", "postLayerName"] + conn_vars
conn_default =  ["",""] + conn_default
conn_ignore = ["",""] + conn_ignore

# Hardwire nxScale and nyScale as the starting elements of the Layer lists

layer_vars = ["nxScale", "nyScale", "Layer_X_Dimension", "Layer_Y_Dimension", "nf"] + layer_vars
layer_default =  ["","","","",""] + layer_default
layer_ignore = ["","","","",""] + layer_ignore


# Build dictionaries of connection and layer names:

with open(layer_loc) as f1:
    layer_dict = [line.strip("\n") for line in f1]
with open(conn_loc) as f2:
    conn_dict = [line.strip("\n") for line in f2]
print(layer_dict)
print(conn_dict)

# Create lists of names, types, etc. from the param file:
colNx = 0
colNy = 0

objectFlag = ""
layer_types = []
layer_names = []
layer_content= []
conn_types = []
conn_names = []
conn_content = []

# Multi-regex handler, for readability later. Sequentailly pares down a passed list of regexes.

def findregex(reg, line):
    if isinstance(reg, list):
        for i in reg:
            temp = re.search(i, line)
            if temp:
                line = temp.group()
        return line


with open(param_location, "r") as file:
    for line in file:

# Check if current line is specifying colNx or colNy

            if re.search("nx\s", line):
                colNx = float(re.search("-?\d+(\.\d+)?", line).group())
                continue
            if re.search("ny\s", line):
                colNy = float(re.search("-?\d+(\.\d+)?", line).group())
                continue

# Create two full-content lists of lists, containing all lines in all conn/layer objects

            for i in layer_dict:
                type_regex = "\A" + i + "\s"
                name_regex = ["(?<=" + i + ").+", "\w+"]
                layer = re.search(type_regex, line)
                if layer:

                    layer_types.append(findregex(["\w+(?=\s)"],layer.group()))
                    layer_names.append(findregex(name_regex, line))
                    objectFlag = "Layer"
                    layer_content.append([])

            for i in conn_dict:
                type_regex = "\A" + i + "\s"
                name_regex = ["(?<=" + i + ").+", "\w+"]
                conn = re.search(type_regex, line)
                if conn:
                    conn_types.append(findregex(["\w+(?=Conn)"],conn.group()))
                    conn_names.append(findregex(name_regex, line))
                    objectFlag = "Conn"
                    conn_content.append([])

            if objectFlag == "Layer":
                layer_content[-1].append(line)
            if objectFlag == "Conn":
                conn_content[-1].append(line)

# Go through content lists, extracting desired values and saving them into lists of lists.

layer_values = []
conn_values = []
for i in layer_content:
    layer_values.append([])
    for c,j in enumerate(layer_vars):
        found = False
        for k in i:
            if re.search(j + " ",k):
                val = findregex(["(?<=" + j + ").+", "-?\w+(\.\w+)?"],k)
                found = True
                if val == layer_ignore[c]:
                    layer_values[-1].append("")
                    break
                else:
                    layer_values[-1].append(val)
                    break
            if k == i[-1] and found == False:
                layer_values[-1].append(layer_default[c])

for i in conn_content:
    conn_values.append([])
    for c,j in enumerate(conn_vars):
        found = False
        for k in i:
            if re.search(j + " ",k):
                val = findregex(["(?<=" + j + ").+", "-?\w+(\.\w+)?"],k)
                found = True
                if val == conn_ignore[c]:
                    conn_values[-1].append("")
                    break
                else:
                    conn_values[-1].append(val)
                    break
            if k == i[-1] and found == False:
                conn_values[-1].append(conn_default[c])

# Create list of indexes that say where in the layer list each prelayer and postlayer is.

pre_index = []
post_index = []

for i in conn_values:
    for c,j in enumerate(layer_names):
        if i[0] == j:
            pre_index.append(c)
            continue
        if i[1] == j:
            post_index.append(c)
            continue

# Check to make sure conn names follow conventions suggested by pre- and post- layers

conn_name_mismatch = [True]*len(conn_names)
for i,j in zip(pre_index,post_index):
    for c,k in enumerate(conn_names):
        if k == layer_names[i] + "To" + layer_names[j]:
            conn_name_mismatch[c] = False

# Calculate layer dimensions, put them on the front of the string
#print(layer_names)
#print(layer_types)
#print(conn_names)
#print(conn_types)
#print(layer_vars)
#for i in layer_values:
#    print(i)
for i in layer_values:
    i[2] = str(float(i[0])*colNx)
    i[3] = str(float(i[1])*colNy)



############################################################
################## DETERMINE PVP OUTPUTS  ##################
############################################################


sparse_layer = []
error_layer = []
error_input = []
sparse_to_error = []
recon_name = []
ccx = conn_vars.index("channelCode")

for c,i in enumerate(layer_names):
    if layer_types[c] == "HyPerLCALayer" or layer_types[c] == "ISTALayer":
        for pre,post,d in zip(pre_index,post_index,range(0, len(conn_names))):
            print(layer_names[pre] + " " + layer_names[post])
            if layer_names[pre] == i:
                if re.search("Error",layer_types[post]) or re.search("Mask",layer_types[post]):
                    if conn_values[d][ccx] != '0':
                        error_layer.append(layer_names[post])
                        sparse_layer.append(layer_names[pre])
                        sparse_to_error.append(conn_names[d])

for i in error_layer:
    for pre,post,d in zip(pre_index,post_index,range(0,len(conn_names))):
        if layer_names[post] == i and conn_values[d][ccx] == '0':
            error_input.append(layer_names[pre])

for c,i in enumerate(layer_names):

    if (re.search("Recon",i) or re.search("recon",i)):
        recon_name.append(i)

hyper = []
err = []
input = []
w = []
recon = []
masterlist = []

for i in sparse_layer:
    hyper.append("_" + i + ".pvp")
    masterlist.append(hyper[-1])
for i in error_layer:
    err.append("_" + i + ".pvp")
    masterlist.append(err[-1])
for i in error_input:
    input.append("_" + i + ".pvp")
    masterlist.append(input[-1])
for i in sparse_to_error:
    w.append("_" + i + ".pvp")
    masterlist.append(w[-1])
for i in recon_name:
    recon.append("_" + i + ".pvp")
    masterlist.append(recon[-1])

print(hyper)
print(err)
print(input)
print(w)
print(recon)


actual_pvp = []
for i in listdir(output_dir):
    if re.search(".pvp", i):
        actual_pvp.append(i)
    if os.stat(output_dir + "/" + i).st_size == 0:
        print("Warning: " + i + " is an empty file.")
    for c,j in enumerate(hyper):
        if re.search(j, i):
            hyper[c] = i
    for c,j in enumerate(err):
        if re.search(j, i):
            err[c] = i
    for c,j in enumerate(input):
        if re.search(j, i):
            input[c] = i
    for c,j in enumerate(w):
        if re.search(j, i):
            w[c] = i
    for c,j in enumerate(recon):
        if re.search(j, i):
            recon[c] = i
    for c,j in enumerate(masterlist):
        if re.search(j, i):
            masterlist[c] = i

for i in masterlist:
    if re.search("^_", i):
        print("Warning: " + i + " not found")

for i in actual_pvp:
    if not i in masterlist:
        print("Warning: did not parse \"" + i + "\" found in output file")

f = open("found_pvps.txt", "w")
for c,i in enumerate(hyper):
    if i[0] == "_":
        f.write("")
    elif os.stat(output_dir + "/" + i).st_size == 0:
        f.write("")
    else:
        f.write(i)
    if c != len(hyper)-1:
        f.write(",")
f.write("\n")


for c,i in enumerate(err):
    if i[0] == "_":
        f.write("")
    elif os.stat(output_dir + "/" + i).st_size == 0:
        f.write("")
    else:
        f.write(i)
    if c != len(err)-1:
        f.write(",")
f.write("\n")

for c,i in enumerate(input):
    if i[0] == "_":
        f.write("")
    elif os.stat(output_dir + "/" + i).st_size == 0:
        f.write("")
    else:
        f.write(i)
    if c != len(input)-1:
        f.write(",")
f.write("\n")


for c,i in enumerate(w):
    if i[0] == "_":
        f.write("")
    elif os.stat(output_dir + "/" + i).st_size == 0:
        f.write("")
    else:
        f.write(i)
    if c != len(w)-1:
        f.write(",")
f.write("\n")

for c,i in enumerate(recon):
    if i[0] == "_":
        f.write("")
    elif os.stat(output_dir + "/" + i).st_size == 0:
        f.write("")
    else:
        f.write(i)
    if c != len(recon)-1:
        f.write(",")
f.write("\n")


f.close()

print(hyper)
print(err)
print(input)
print(w)
print(recon)
