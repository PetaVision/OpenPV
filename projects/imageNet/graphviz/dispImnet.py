#!/usr/bin/env python

import networkx as nx
import matplotlib.pyplot as plt
import os.path

try:
    from networkx import graphviz_layout
except ImportError:
    raise ImportError("\n\nERROR: This needs Graphviz and either PyGraphviz or Pydot\n\n")

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    f.close()
    return i+1

#wnidFile  = "synsetParentWNIDs.ssv"
#nameFile  = "synsetParentNames.ssv"
#outFile   = "IMNet-Comb.gml"

#wnidFile  = "downloadedWNIDs.ssv"
#nameFile  = "downloadedNames.ssv"
#outFile   = "downloadedGraph.gml"

wnidFile  = "n02084071_Children_WNIDs.ssv"
nameFile  = "n02084071_Children_Names.ssv"
outFile   = "n02084071_Children_Graph.gml"

numLines = 0
if (file_len(wnidFile) != file_len(nameFile)):
    print "\n\nERROR: Files are not the same length!\n\n"
else:
    numLines = file_len(wnidFile)

if os.path.isfile(wnidFile):
    try:
        wnidFH = open(wnidFile,"r")
    except IOError:
        print "\n\nERROR: Can't find input file"
        raise
else:
    print "\n\nERROR: Can't find input file\n\n"

if os.path.isfile(nameFile):
    try:
        nameFH = open(nameFile,"r")
    except IOError:
        print "cant find input file"
        raise
else:
    print "\n\nERROR: Can't find input file\n\n"

G = nx.Graph()

newNodeIdx = 0
seenList = []
edgeList = []
for lineIdx in range(1,numLines+1): #range(a,b) returns list from a to b-1
    wnidLine = wnidFH.readline().rstrip()
    nameLine = nameFH.readline().rstrip()

    wnidArry = wnidLine.split(";")
    nameArry = nameLine.split(";")

    if (len(wnidArry) != len(nameArry)):
        print "\n\nERROR: Line array lengths are not equal.\nlineIdx = ",lineIdx,"\n"
        print "Length of wnidArry = ",len(wnidArry),"\tLength of nameArry = ",len(nameArry),"\n\n"

    for i in range(0,len(wnidArry)): #For each item in the input line (should be all parents of a particular leaf node)
        #add node if new
        if wnidArry[i] not in seenList: #if we have not seen this node before
            seenList.append(wnidArry[i]) #seenList index = G.node index = newNodeIdx
            G.add_node(newNodeIdx, label=wnidArry[i], words=nameArry[i])
            #add connection
            if i > 0: #if this isn't the first element in the input line
                for j in range(0,newNodeIdx): #parse through graph to find the index of the parent node
                    if G.node[j]['label'] == wnidArry[i-1]: #if we have found the parent node
                        edgeList.append((j,newNodeIdx)) #add parent node index to just added node
                        break
            newNodeIdx += 1 #idx of next node to be added

G.add_edges_from(edgeList) #should add all the edges, ignoring duplicates

print "\nNum nodes\tNum edges"
print G.number_of_nodes(),"\t\t",G.number_of_edges()
print "\nNum lines:",numLines

nx.write_gml(G,outFile) #Write to a GML (Graph Markup Language) file
