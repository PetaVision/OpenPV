from os import listdir
import re
import sys
import math

############################################################
######################## INPUT #############################
############################################################

# Input files and sources for connection/layer name:
param_location = (sys.argv[1])
llines = listdir(sys.argv[2] + "/src/layers")
clines = listdir(sys.argv[2] + "/src/connections")

# Colorby determines should be set to 'dimension', 'phase', or 'type':
colorby = "phase"
line_colorby = "channelCode"
line_boldby = "plasticityFlag"
line_types = ["0", "1"]
line_colorcode = ["0f0","f00"]

# Color ranges for coloring layer shapes
rmin = "77"
rmax = "dd"
gmin = "77"
gmax = "dd"
bmin = "99"
bmax = "ff"

# Lists of variables to look for in each layer and conn, plus defaults if they are not found.

layer_vars = ["originalLayerName","phase"]
conn_vars = ["channelCode","maskLayerName","plasticityFlag","delay","originalConnName","weightInitType","initWeightsFile","pvpatchAccumulateType"]
layer_default = ["",""]
conn_default = ["","","","","","","",""]
layer_ignore = ["",""]
conn_ignore = ["","","","0.000000","","NULL","",""]


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

layer_dict = []
conn_dict = []

for i in llines:
    j = re.search(".+(?=cpp)", i)
    if j:
        k = re.search("\w+", j.group())
        if k:
            layer_dict.append(k.group())
for i in clines:
    j = re.search(".+(?=cpp)", i)
    if j:
        k = re.search("\w+", j.group())
        if k:
            conn_dict.append(k.group())

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

# Ignore lines commented with //

        if re.search("^\s+\/\/",line):
            continue

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
        for d,k in enumerate(i):
            if re.search(j + " ",k):
                val = findregex(["(?<=" + j + ").+", "-?\w+(\.\w+)?"],k)
                found = True
                if val == layer_ignore[c]:
                    layer_values[-1].append("")
                    break
                else:
                    layer_values[-1].append(val)
                    break
            if d == len(i)-1 and found == False:
                layer_values[-1].append(layer_default[c])

for i in conn_content:
    conn_values.append([])
    for c,j in enumerate(conn_vars):
        found = False
        for d,k in enumerate(i):
            if re.search(j + " ",k):
                #START INITWEIGHTSFILE KLUGE
                if j == "initWeightsFile":
                    val = re.findall("(?<=/)[\w\.]+",k)[-1]
                #END INITWEIGHTSFILE KLUGE
                else: 
                    val = findregex(["(?<=" + j + ").+", "-?\w+(\.\w+)?"],k)
                found = True
                if val == conn_ignore[c]:
                    conn_values[-1].append("")
                    break
                else:
                    conn_values[-1].append(val)
                    break
            if d == len(i)-1 and found == False:
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
#
# Do NOT flag a name mismatch in the following sitution:  
#  prelayer:  a_x
#  postlayer: b_x
#  connection name: aTob_x
#
conn_name_mismatch = [True]*len(conn_names)
for i,j in zip(pre_index,post_index):
    for c,k in enumerate(conn_names):
        if k == layer_names[i] + "To" + layer_names[j]:
            conn_name_mismatch[c] = False
            break
        prel = re.search("[^_]+",layer_names[i])
        predim = re.search("(?<=_).+",layer_names[i])
        postdim = re.search("(?<=_).+",layer_names[j])
        if prel and predim and postdim:
            if predim.group() == postdim.group():
                if k == prel.group() + "To" + layer_names[j]:
                    conn_name_mismatch[c] = False
                    break

# Calculate layer dimensions, put them on the front of the string
print(layer_names)
print(layer_types)
print(conn_names)
print(conn_types)
print(layer_vars)
for i in layer_values:
    i[2] = str(float(i[0])*colNx)
    i[3] = str(float(i[1])*colNy)
for i in layer_values:
    print(i)
for i in conn_values:
    print(i)
############################################################
######################## COLORING ##########################
############################################################

color_value = []
color_code = []
nocolorflag = False

# Determine which value is to be colored by, store by layer in a color value list

if colorby == "dimension":
    for i in layer_values:
            color_value.append(math.log(float(i[2])*float(i[3]),2))
elif colorby == "type":
    for i in layer_types:
        pass
elif colorby in layer_vars:
    for i in layer_values:
        color_value.append(float(i[layer_vars.index(colorby)]))
        

# Generate 6-digit hex color codes

rmin = int(rmin, 16)
rmax = int(rmax, 16)
gmin = int(gmin, 16)
gmax = int(gmax, 16)
bmin = int(bmin, 16)
bmax = int(bmax, 16)


clo = min(color_value) 
chi = max(color_value)
rcodes = []
gcodes = []
bcodes = [] 
for i in color_value:
    if rmin == rmax:
        rcodes.append(hex(rmin).split('x')[1])
    else:
        t = int(rmin + (((i - clo)*(rmax - rmin))/(chi - clo))) 
        rcodes.append(hex(t).split('x')[1])
    if gmin == gmax:
        gcodes.append(hex(gmin).split('x')[1])
    else:       
        t = int(gmin + (((i - clo)*(gmax - gmin))/(chi - clo)))
        gcodes.append(hex(t).split('x')[1])
    if bmin == bmax:
        bcodes.append(hex(gmin).split('x')[1])
    else:
        t = int(bmin + (((i - clo)*(bmax - bmin))/(chi - clo)))
        bcodes.append(hex(t).split('x')[1])

# Clean up for too many/too few digits

for i in range(0, len(color_value)):
    if len(rcodes[i]) < 2:
        rcodes[i] = '0' + rcodes[i]
    elif len(rcodes[i]) > 2:
        rcodes[i] = rcodes[i][0:2]
    if len(gcodes[i]) < 2:
        gcodes[i] = '0' + gcodes[i]
    elif len(gcodes[i]) > 2:
        gcodes[i] = gcodes[i][0:2]
    if len(bcodes[i]) < 2:
        bcodes[i] = '0' + bcodes[i]
    elif len(bcodes[i]) > 2:
        bcodes[i] = bcodes[i][0:2]

# Save codes to color_code
for r,g,b in zip(rcodes,gcodes,bcodes):
    color_code.append(r + g + b)

############################################################
#################### MERMAID OUTPUT ########################
############################################################

# Objects and links
f = open("param_graph", "w")
f.write("graph BT;\n")
op,cl = "[","]"
for c,i in enumerate(layer_names):
    op,cl = "[","]"
    if layer_types[c] == "HyPerLCALayer":
        op,cl = "{","}"
    f.write(i + op + i + "<br><i>" + layer_types[c] + "</i><br>")
    f.write(str(int(float(layer_values[c][2]))) + ":" + str(int(float(layer_values[c][3]))) + ":" + layer_values[c][4])
    if len(layer_values[c]) == 5:
        continue
    else:
        f.write("<br>")
        for j in range(5, len(layer_vars)):
            if layer_values[c][j] == "":
                continue
            else:
                f.write(layer_vars[j] + ": " + layer_values[c][j])
            if j != len(layer_vars)-1:
                f.write("<br>")
    f.write(cl + ";\n")

for pre,post,c in zip(pre_index,post_index,range(0, len(conn_names))):
    f.write(layer_names[pre] + "-->|" + conn_types[c])
    if len(conn_vars) == 2:
        continue
    else:
        f.write("<br>")
        # To display less information about Conns, change the starting value
        # of the loop below from "2" to "5"
        for j in range(2, len(conn_vars)):
            if conn_values[c][j] == "":
                continue
            else:
                if conn_vars[j] == "initWeightsFile" or conn_vars[j] == "pvpatchAccumulateType":
                    f.write(conn_values[c][j])
                else:
                    f.write(conn_vars[j] + ": " + conn_values[c][j])
            if j != len(conn_vars)-1:
                f.write("<br>")
    f.write("|" + layer_names[post] + ";\n")

# Line color

indb = conn_vars.index(line_boldby)
indc = conn_vars.index(line_colorby)
if line_boldby in conn_vars and line_colorby in conn_vars:
    # here
    for c,i in enumerate(conn_values):
        if i[indb] == "true":
            s = "4"
        else:
            s = "2"
        alreadywrote = False
        for d,j in enumerate(line_types):
            if i[indc] == j:
                f.write("linkStyle " + str(c) + " fill:none,stroke:#" + line_colorcode[d] + ",stroke-width:" + s + "px")
                alreadywrote = True
                if conn_name_mismatch[c] == True:
                    f.write(",stroke-dasharray: 5, 5;\n")
                else:
                    f.write(";\n")
        if alreadywrote == False:
            f.write("linkStyle " + str(c) + " fill:none,stroke:#000000,stroke-width:" + s + "px")
            if conn_name_mismatch[c] == True:
                f.write(",stroke-dasharray: 5, 5;\n")
            else:
                f.write(";\n")
# Object color
for i in color_code:
    f.write("classDef color" + i + " fill:" + i +  ",stroke:#333,stroke-width:2px;\n")
for i,j in zip(layer_names,color_code):
    f.write("class " + i + " color" + j + ";\n")
