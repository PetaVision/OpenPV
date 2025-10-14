'''
Public facing general purpose analysis and data handling tools for work with Petavision. Extends the functionality of existing pvtools
'''
import pvtools as pv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Any, Dict, Union
from collections import defaultdict
import argparse
import sys


def makeListFromFile(path):
    tempFile=open(path,'r')
    tempbuffer=tempFile.read()
    tempFile.close()
    return tempbuffer.strip().split('\n')

def findBetween(substring,startExpression,endExpression):
    '''
    Finds a subtring between two existing substring within a string

    **Parameters**

        substring:  *str*
            The string to search inside
        
        startExpression:    *str*
            For left to right languages, a substring or character representing the left most bound of the substring to find, first match only

        endExpression:  *str*
            For left to right languages, a substring or character representing the right most bound of the substring to find, first match only

    **Returns**

        If found the substring as *str*, if not found, an empty string as *str*
    '''
    try:
        start = substring.index(startExpression) + len(startExpression)
        end = substring.index(endExpression, start)
        return substring[start:end]
    except ValueError:
        return ""

#extract petavison generated lua parameters from a file and neatly organize them, params file path must be a .params.lua file
#Author: Nick Bruns
#Created: 4/23/2025
def extractParameters(paramsFilePath:str,
                      startKeyword:str = "local pvParameters = {",
                      endKeyword:str = "} --End of pvParameters",
                      layerEndKeyword:str="};\n\n",
                      nonLUAinclusiveStartKeyword:str="HyPerCol",
                      removeQuoteCharacters:bool=True,
                      usePythonicTypes:bool=True) -> Dict[str, Dict[str, Any]]:

    '''
    Traverses a petavision output run folder, identifying the input layers that were used in the network, then outputing two dictionaries corresponding to the

    **Parameters**

        paramsFilePath:  *str*, *path*
            A path to a .params or .params.lua petavision generated parameter file the describes the layout of a petavision network

    **Returns**
        A nested *dict()* object that contains each component of the network
        
    '''

    #default assumption is that the layout follows a lua standard    
    LUA_LAYOUT=True

    #detect if a params files or a params.lua file, default assumption is params.lua
    if paramsFilePath.endswith(".params"):
        LUA_LAYOUT=False
    
    #define important values I want to replace
    FALSE_STR_REP='false'
    TRUE_STR_REP='true'
    NONE_STR_REP='NULL'
    INFINITY_STR_REP='infinity'
    LIST_START_CHAR='{'
    LIST_END_CHAR='}'
    LAYER_TYPE_KEY='groupType'
    
    #check if a string could be converted to a numeric type
    def is_numeric(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    paramsFile=open(paramsFilePath,'r')
    paramsFileBuffer=paramsFile.read()
    paramsFile.close()

    #if the layout is LUA standard then use the start stop keywords to extract valid text
    if LUA_LAYOUT:
        #find just the characters between the start and end keywords, then strip leading and trailling whitespace,
        # then remove the final two characters which are uncaught "}; characters "
        parameterRawString=paramsFileBuffer[paramsFileBuffer.find(startKeyword)+len(startKeyword):paramsFileBuffer.find(endKeyword)].strip()[0:-2]
    #if the layout is not LUA standard, then it should assumed that the parameter data starts at the HyPerCol definition and ends at EOF
    else:
        parameterRawString=paramsFileBuffer[paramsFileBuffer.find(nonLUAinclusiveStartKeyword):None].strip()[0:-2]

    #next split into descrete text segments, one for each layer
    layerTextSegments=[layerData.strip() for layerData in parameterRawString.split(layerEndKeyword)]

    # #buffer lists to hold the splits
    # layerNames=[]
    # layerText=[]
    # layerLines=[]

    #master dictionary to hold all layers
    masterDict=dict()

    #if I cannot reuse dictionaries as a buffer, I have to define here and make copies of as needed
    # tempLayerDictionary={}

    for i,layer in enumerate(layerTextSegments):

        #print(layer)

        #if lua style layout, can always guaruntee a recursive split on equals
        if LUA_LAYOUT:
            tempLayerName,tempLayerText=[pair.strip() for pair in layer.split("=",maxsplit=1)]
        #if not a LUA layout the layer type then layer name preceeds the the brackets containing the text 
        else:
            tempLayerTypeAndName,tempLayerText=[pair.strip() for pair in layer.split("=",maxsplit=1)]
            #if instructed to replace quote characters, do so now
            if removeQuoteCharacters:
                #replace double quotes with empty string
                tempLayerTypeAndName=tempLayerTypeAndName.replace("\"","")

            #split the the layer name and type on a single whitespace character in the non-LUA standard, layer type preceeds the layer name
            tempLayerType,tempLayerName = [pair.strip() for pair in tempLayerTypeAndName.split(" ")]

        # layerNames.append(tempLayerName)
        # layerText.append(tempLayerText)
        #all layer text begins with a { character which needs to be cut out using [1:None] slicing prior to any striping leading and trailing whitespace
        # after that is is done, entries can be split on the newline character and the leading and trailing whitespace can be removed once more
        # then a slice of [0:-1] removes the semicolon at the end of each entry
        tempLayerLines=[line.strip()[0:-1] for line in tempLayerText[1:None].strip().split('\n')]
        # layerLines.append(tempLayerLines)

        #might need to clear dictionary to reuse as a buffer safely
        # tempLayerDictionary.clear()

        tempLayerDictionary=dict()

        #if not a lua Layout, then layer type needs to be added to the temp dictionary    
        if LUA_LAYOUT is False:
            tempLayerDictionary.update({LAYER_TYPE_KEY:tempLayerType})

        #loop over each line in the layer
        for j,line in enumerate(tempLayerLines):

            # #error preemption to read an non field-key covertible end character
            #NOTE: not needed after finding that the unreadable end characters exist in both formats
            # if (LUA_LAYOUT is False) and i==(len(layerTextSegments)-1) and j==(len(tempLayerLines)-1):
            #     break

            #seperate the field value pair by splitting on equals sign and stripping whitespace
            tempField,tempValue=[pair.strip() for pair in line.split("=")]

            #if instructed to remove double quotes
            if removeQuoteCharacters is True:
                #remove the double quote characters and replace with nothing
                tempValue=tempValue.replace("\"","")

            #if pythonic type use is specified, attempt to cast strings to types
            if usePythonicTypes is True:
                #if tempValue could be a numeric type, true is returned and it is cast as a float
                if is_numeric(tempValue):
                    tempValue=float(tempValue)
                #if the start and end character of tempValue are brackets, this indicates a list
                #it is assumed the list holds numeric values
                elif tempValue.startswith(LIST_START_CHAR) & tempValue.endswith(LIST_END_CHAR):
                    #remove the brackets from tempValue, 
                    # then split into an array seperated by a comma,
                    #then use list comprehension to covert each value to a float
                    #finally reassign temp value
                    tempValue=[float(element) for element in tempValue.replace(LIST_START_CHAR,"").replace(LIST_END_CHAR,"").split(',')]
                #if infinity, then use python inifinity
                elif tempValue == INFINITY_STR_REP:
                    tempValue=float('inf')
                elif tempValue == TRUE_STR_REP:
                    tempValue = True
                elif tempValue == FALSE_STR_REP:
                    tempValue = False
                elif tempValue == NONE_STR_REP:
                    tempValue = None
                else:
                    pass
                
            tempLayerDictionary.update({tempField:tempValue})

        #may need to copy tempLayerDictionary here

        #update master dictionary
        masterDict.update({tempLayerName:tempLayerDictionary})

        # may need to clear tempLayerDictionary here
        #tempLayerDictionary.clear()
    
    return masterDict

def vertical_layout(G, node_order, layer_spacing=1.5, horizontal_spacing=2.0):
    """
    Generate a position dictionary for a vertical layout, with nodes in order from top to bottom.
    
    Parameters:
        G (networkx.Graph): Your graph.
        node_order (dict): Mapping of node -> vertical layer (0 = top).
        layer_spacing (float): Vertical distance between layers.
        horizontal_spacing (float): Horizontal spread between nodes in the same layer.
    
    Returns:
        dict: Node positions for drawing.
    """
    # from collections import defaultdict

    # Group nodes by layer
    layers = defaultdict(list)
    for node, order in node_order.items():
        layers[order].append(node)

    pos = {}
    for layer, nodes_in_layer in sorted(layers.items()):
        y = -layer * layer_spacing  # Flip so 0 is at the top
        count = len(nodes_in_layer)
        x_positions = [
            horizontal_spacing * (i - (count - 1) / 2.0) for i in range(count)
        ]
        for x, node in zip(x_positions, nodes_in_layer):
            pos[node] = (x, y)
    return pos

#draw a network diagram given a parameter path
#Author: Nick Bruns
#Created: 5/9/2025
def drawNetwork(parameterFilePath,connectionLabelStyle='terse',includeOriginalConnectionName:bool=True,KKRedraw:bool=True):

    validStyles = {'terse', 'verbose'}
    if connectionLabelStyle not in validStyles:
        raise ValueError("connectionLabelStyle variable must be one of %r." % validStyles)

    #for added flexibility, the parameter file path could be an already computed dict for reasons such as renaming
    if isinstance(parameterFilePath,str):
        #extract parameters as a dictionary
        parameters=extractParameters(parameterFilePath,removeQuoteCharacters=True,usePythonicTypes=True)
    elif isinstance(parameterFilePath,dict):
        parameters=parameterFilePath
    else:
        raise ValueError(f"Argument 1 / parameterFilepath must be a path to a parameter file or parameter dictionary itself, not {type(parameterFilePath)}")

    #definition of some various formating keywords
    momentumConnLineStyle='dashed'
    defaultLineStyle='solid'
    defaultNodeColor='lightgreen'
    cloneConnLineStyle=':'
    arrowShape='-|>'
    shortHandCloneSymbol="W (clone)"
    shortHandMommentumSymbol="W"
    shortHandIdentSymbol="A"
    shortHandTransposeSymbol="T"

    #determine what layers there are
    layerRequiredKey='phase'
    layerNames=[]
    layerPhase=[]
    layerType=[]
    for topLevelKey in parameters.keys():
        if layerRequiredKey in parameters[topLevelKey]:
            #append the information i need to lists
            layerNames.append(topLevelKey)
            layerPhase.append(int(parameters[topLevelKey]['phase']))
            layerType.append(parameters[topLevelKey]['groupType'])

            print(f"{topLevelKey}\t{int(parameters[topLevelKey]['phase'])}")
            
    #determine what connections I have
    connectionRequiredKeys=['preLayerName','postLayerName']
    connectionNames=[]
    connectionType=[]
    connectionPre=[]
    connectionPost=[]
    connectionOriginalConnName=[]
    for topLevelKey in parameters.keys():
        #use list comprehension to run through every key that should exist in a connection
        #then evaluate using 'in' to get a boolean for if the key is present
        #then use the all() to test if the boolean list is all true
        if all([key in parameters[topLevelKey] for key in connectionRequiredKeys]):
            connectionNames.append(topLevelKey)
            connectionType.append(parameters[topLevelKey]['groupType'])
            connectionPre.append(parameters[topLevelKey]['preLayerName'])
            connectionPost.append(parameters[topLevelKey]['postLayerName'])
            #the use of get here preempts any situation where there is no originalConnName key
            tempOrigConnName=parameters[topLevelKey].get('originalConnName')
            #if the value is none then replace with noncharacter space
            if tempOrigConnName is None:
                tempOrigConnName = ''
            connectionOriginalConnName.append(tempOrigConnName)  
            print(f"{topLevelKey}\t\t{(parameters[topLevelKey]['preLayerName'])}\t{(parameters[topLevelKey]['postLayerName'])}")
    #somewhat vestigial operation to calculate horizontal offsets of network layers

    #set removes duplicates
    uniquePhases=list(set(layerPhase))
    #use list comprehension to count the number of occurrances of each phase
    phaseCounts=[layerPhase.count(phase) for phase in uniquePhases]
    #make an empty list of zeros for horizontal position offsets
    relativeHorizontalPosition=[0 for i in range(len(layerPhase))]
    # Calculate unique horizontal offsets for each layer
    phaseCountOffsetBuffer=phaseCounts
    for i,phase in enumerate(layerPhase):
        #get the index in the unique phase list and use it to find a position in a count offset buffer
        bufferPosition=uniquePhases.index(phase)
        # then use this index to find the location in a count buffer and decrement it by one
        phaseCountOffsetBuffer[bufferPosition]=phaseCountOffsetBuffer[bufferPosition]-1
        #finally, add the offset to the position at that location
        relativeHorizontalPosition[i]=relativeHorizontalPosition[i]+phaseCountOffsetBuffer[bufferPosition]

    ## End vestigial operation

    verboseConnectionEdgeLabels=dict()
    connectionStyleDict=dict()
    terseConnectionEdgeLabels=dict()
    copiedConnectionDict=dict()

    #loop through connections, updating various style dictionaries as needed
    for name,pre,post,connType,origConnName in zip(connectionNames,connectionPre,connectionPost,connectionType,connectionOriginalConnName):
        #add to the connection label dictionary
        verboseConnectionEdgeLabels.update({(pre,post):f"{name}\n{connType}"})

        #update the dict for connections with duplicated outputs
        copiedConnectionDict.update({(pre,post):origConnName})

        #update style dictionary for setting connection properties
        if connType == 'MomentumConn':
            # a dict has to store this data to work
            connectionStyleDict.update({(pre,post):momentumConnLineStyle})
        elif connType == 'CloneConn':
            connectionStyleDict.update({(pre,post):cloneConnLineStyle})
        else:
            connectionStyleDict.update({(pre,post):defaultLineStyle})
        
        #update dictionary of shorthand names
        if connType == 'MomentumConn':
            terseConnectionEdgeLabels.update({(pre,post):shortHandMommentumSymbol})
        elif connType == 'CloneConn':
            terseConnectionEdgeLabels.update({(pre,post):shortHandCloneSymbol})
        elif connType == 'IdentConn':
            terseConnectionEdgeLabels.update({(pre,post):shortHandIdentSymbol})
        elif connType == 'TransposeConn':
            terseConnectionEdgeLabels.update({(pre,post):shortHandTransposeSymbol})
        else:
            terseConnectionEdgeLabels.update({(pre,post):''})


        
    #initialize a vertical and horizontal positioning dict, and various style tracking dicts such as for color or label
    verticalNodeOrder={}
    horizontalNodePositioning={}
    nodeStyle={}

    # Loop over each layer and update as needed
    for name,phase,type,column in zip(layerNames,layerPhase,layerType,relativeHorizontalPosition):
        # Update a somewhat vestigial positioning dict
        horizontalNodePositioning.update({name:(column,phase)})
        # Update a node ordering dict
        verticalNodeOrder.update({name:phase})
        # Update a type dict
        nodeStyle.update({name:type})

    #determine what sort of labelling to use
    if connectionLabelStyle == "terse":
        connectionLabelDisplay = terseConnectionEdgeLabels
    elif connectionLabelStyle == "verbose":
        connectionLabelDisplay = terseConnectionEdgeLabels
    else:
        connectionLabelDisplay = None

    #if instructed to label the original connections, handle accordingly
    if includeOriginalConnectionName is True and connectionLabelDisplay is not None:
        for pre, post in zip(connectionPre,connectionPost):
            #extract existing label from dict
            leadingLabel=connectionLabelDisplay[(pre,post)]
            #update the key with the new field value 
            connectionLabelDisplay.update({(pre,post):f"{leadingLabel}\n{copiedConnectionDict[(pre,post)]}"})
    elif includeOriginalConnectionName is True and connectionLabelDisplay is None:
        connectionLabelDisplay = copiedConnectionDict
    else:
        pass

    #NOTE: plotting section
    # Create a directed graph
    G = nx.DiGraph()

    #zip together pre and post connections as a list of tuples, assign as directed edges, this may need some sorting, actually maybe not
    edges = list(zip(connectionPre,connectionPost))
    G.add_edges_from(edges)

    #add nodes with ordering
    G.add_nodes_from(verticalNodeOrder.keys())

    # # Use the vertical layout function
    pos = vertical_layout(G, verticalNodeOrder)

    if KKRedraw:
        #after computing a vertical layout, then process through a Kamada Kawai model to attempt reduction in overlap
        pos=nx.kamada_kawai_layout(G,pos=pos)
    #nx.layout

    # Draw square-shaped nodes
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color=defaultNodeColor, node_size=1000, node_shape='s')

    # Draw labels on nodes
    nx.draw_networkx_labels(G, pos, font_size=8)

    # Draw the edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=connectionLabelDisplay, font_color='red', font_size=8,verticalalignment='center',rotate=False)

    # Draw curved edges with arrows
    for i,edge in enumerate(G.edges()):
        nx.draw_networkx_edges(
            G, pos,
            style=connectionStyleDict[edge],    #drawing does not happen sequentially so a dict has to store everything
            edgelist=[edge],
            width=3,
            arrowstyle=arrowShape,
            arrowsize=30,
            connectionstyle='arc3,rad=0.2'
        )

    plt.title(parameterFilePath)
    plt.axis('off')
    #plt.legend()
    plt.show()

#to use commands from the command line, just type in the path to your python interpretter, the path of this .py file, the function name and then the arugments to give it
if __name__=="__main__":
    globals()[sys.argv[1]](sys.argv[2])