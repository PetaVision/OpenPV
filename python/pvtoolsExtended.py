'''
Public facing general purpose analysis and data handling tools for work with Petavision. Extends the functionality of existing pvtools

Author: Nick Bruns
'''
import pvtools as pv
import numpy as np
# import matplotlib
# matplotlib.use('ps')
import matplotlib.figure
import matplotlib.text
import matplotlib.pyplot as plt
import networkx as nx
from typing import Any, Dict, Union
from collections import defaultdict
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import argparse
import sys
import itertools
import math
import os

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

def formatStringWithSeperators(s,insertionChar='\n'):
    '''
    Given any string with any type of seperator: (camel case, commas, spaces, hypens, etc.) instert an extra character between these breaks
    '''
    #get indices that have a case different from the one that preceded it
    discontinousCaseIndex=np.array([i for i in range(1, len(s)) if s[i].isupper() != s[i-1].isupper()])
    #case changes that I care about occur every second index, thus I want to only insert characters into a new string at those locations
    caseChangeIndex=discontinousCaseIndex[1::2]

    sReturn=s
    for i,insertionPoint in enumerate(caseChangeIndex):
        sReturn=sReturn[:insertionPoint+i]+insertionChar+sReturn[insertionPoint+i:]

    return sReturn

#a small method that unpacks a sparse, non-weight PVP file into a dense one and returns a 4D numpy array
def unpackSparsePVP(input,partialRead:bool=True,blockLength:int=None):
    '''
    This function takes a sparse input/recon pvp file that has already
    been read into an indexed dictionary and returns a 4D numpy array
    matching the original value field of dense PVPs like the following:
        [blockIndex][y][x][nf]

        **Parameters**
            input: *dict()*
                a PVP file that was read in
            partialRead: *bool*
                Boolean indicating if the Input pvp file was read with a start stop frame index defined.
                If **True** (*default*), nb=1 always. If *False* nb=input['header']['nb'].
            blockLength: *int*
                Integer indicating the block length dimension of a sparse PVP file that has had an incomplete range of it partially read. Defines internal 'nb' variable for reshaping a sparse array to a dense array. Default = *None*

        **Returns**
            output: *4Dndnumpy array*
                A dense 4D numpy array matching the format:
                    [nb][y][x][nf]
    '''
    
    if input['header']['filetype']==6:
        nx=input['header']['nx']
        ny=input['header']['ny']
        nf=input['header']['nf']
        if partialRead is True:
            if blockLength:
                #if the pvp has been partially read externally as a range, then use the external one
                nb=blockLength
            else:
                nb=1
        else:
            nb=input['header']['nb']
        pvpShape=(nb,ny,nx,nf)
        output=np.reshape(input['values'].toarray(),pvpShape)
    else:
        print("Input is not sparse")
        output=input['values']

    return output

def previewPVP(PVPname,timeframe,weightDisplayTimestep:int=0,dpiScale:int=30,dpi:int=None,forceResolution:tuple=None,xSlice:slice=None,readMinimal:bool=False,returnInput:bool=False,unravelDims:tuple=None):
    '''
    Predominantly intended for using in jupyter notebooks or interactive windows. Displays a visual representation of a PVP file to the user.
    
    Takes either a path to a PVP file, a dictionary of an already read pvp file, or a 4D numpy array; as well as a timestep/frame/band/block (``nb``) in that file to plot.

    4D PVP dense activity values are stored as
    ``[nb][y][x][nf]``

    6D PVP weight values are stored as
    ``[nb][arbor][kernel][y][x][nf]``

    **Parameters**

        PVPname: *str* or *dict* or *np.ndarray*
            a PVP file to plot
        timeframe: *int*
            Time step in the PVP to plot, corresponds to the ``[nb]`` dimension.
            ``[nb]`` is the which is the first 
        readMinimal: *bool* (Default=False)
            Boolean indicating if the input pvp file is to be partially read with a start stop frame index defined. Set to true for speed.
            If **True** (*default*), nb=1 always. If *False* nb=input['header']['nb'].
        unravelDims: *tuple* (Default=None)
            A 3 element tuple indicating the dimensions ``(ny,nx,nf)`` that the input values should be reshaped into. This operation is useful for use cases where the data being trained on has been reshaped into different vectors or dimensions than what it is normally viewed as naturally.
                **Examples:** Input has a time vector of length 32 and a feature vector of length 22, each dimension should be learned on seperately from the other.
                    
                [nb] => different samples
                [ny] => 1 (not used)
                [nx] => 320 (time)
                [nf] => 22 (features)

                ``unravelDims=(22,320,1)``
                    (samples,1,320,22) ==> (samples,22,320,1)

                **Examples:** Input is a standard video and has an x and y dimension of 128x64, respectively. It is grayscale colored so its feature vector is length 1. The perframe pixel data is reshaped into 1D vector with the x and y dimension flattened into the feature dimension so it can be correlated to other data factors.

                [nb] => frames
                [ny] => 1
                [nx] => 1
                [nf] => 8192 (flattened x and y pixel values)

                ``unravelDims=(64,128,1)``
                    (frames,1,1,8192) ==> (frames,64,128,1)

    **Returns**
        output: optional *4D np.ndnumpy*
            A dense 4D numpy array matching the format:
                [nb][y][x][nf]
    '''
    if(weightDisplayTimestep!=0):
        print("weightDisplayTimestep has been changed from its default value but is being deprecated.\nThe \'timeframe\' parameter will perform the same function for weight files now.\nThe internal timeframe argument will be changed to the user's value for \'weightDisplayTimestep\'.")
        timeframe=weightDisplayTimestep
    #a flag that is set to false which will trigger a specific plotting behavior for weights only
    isWeightFile=False
    # a flag to trigger on the appearance of a sparse file
    isSparseFile=False
    if(isinstance(PVPname,str)):
        print("Filepath to PVP given")

        if readMinimal is True:
            print("readMinimal is True, timeframe variable will be internally set to 0 after pvp file is read")
            pvpInput=pv.readpvpfile(PVPname,lastFrame=timeframe+1,startFrame=timeframe)
            #if I have set to only read a specific given frame, then there is only one timeframe remaining after the targeted read, 0
            timeframe=0
            #nb is always 1 if a minimal read is performed because there is only 1 time point
            nb=1
        else:
            pvpInput=pv.readpvpfile(PVPname)
            #nb is defined as the number of bands
            nb=pvpInput['header']['nbands']

        # this set of variables is exclusively for use by sparse files, 
        # since nb can change if I am doing a minimal read however, I will need to define it near the top
        nx=pvpInput['header']['nx']
        ny=pvpInput['header']['ny']
        nf=pvpInput['header']['nf']

        #nb=pvpInput['header']['nbands']    #nb is always 1 if a minimal read

        #the shape that a sparse array needs to me unpacked into
        pvpShape=(nb,ny,nx,nf)

        #if the filetype field is 4, this must be dense and the values is always of 4D shape 
        if pvpInput['header']['filetype']==4:
            print("Dense 4D PVP detected")
            input=pvpInput['values']
        elif pvpInput['header']['filetype']==5:
            print("6D weight PVP detected. Displaying learning period timestep: %d of %d total timesteps" %(timeframe,pvpInput['values'].shape[0]))
            isWeightFile=True
            #I'm changing the behavior to make it such that timeframe does double duty for weight displayTimestep
            input=pv.arrangedictionary(pvpInput['values'][timeframe,0])
        elif (pvpInput['header']['filetype']==6): #and (pvpInput['header']['datatype']==4):
            print("Sparse PVP activity detected")
            isSparseFile=True

            #in one step, put the sparse values to a dense representation of them in an array and then reshape them to match the expect PVP 4D struct
            input=np.reshape(pvpInput['values'].toarray(),pvpShape)#[timeframe]
            
        else:
            print("PVP file has read successfully but data format case is not recognized, handling as 4D PVP")
            input=pvpInput['values']
    elif(isinstance(PVPname,np.ndarray)):
        print('Generic Numpy nD array given')
        #TODO: add intelligent checking and processing
        input=PVPname
        pvpShape=input.shape
        nb=pvpShape[0]
    else:
        print("Generic nD array PVP given, handling as 4D")
        input=PVPname

    #if unravelDims is not none, then this indicates that a non-default argument has been supplied in the format of an ny,nx,nf shape to refactor the original PVP file into
    if unravelDims:
        print(f"Reshaping pvp from {pvpShape} to {(nb,*unravelDims)}")
        pvpShape = (nb,*unravelDims)
        input = input.reshape(pvpShape)

    #TODO, add a means of checking if a list of paths that should be displayed together

    #if this is a weight file, display it at the timestep given
    if(isWeightFile is True):
        
        plt.figure(1)
        matplotlib.interactive(True)
        #normalize
        wgts_display_8bit = np.uint8(np.squeeze(input) * 127.5 + 127.500001)

        #if the weights cannot be shown in a standard color range
        if(len(wgts_display_8bit.shape)>2) and (min(wgts_display_8bit.shape)>3):
            weightDataStruct=np.moveaxis(wgts_display_8bit,-1,0)
            fig, ax = plt.subplots()
            img = ax.imshow(weightDataStruct[0])

            #allegedly an updating method
            def update(frame):
                img.set_array(frame)
                return [img]

            ani = FuncAnimation(fig, update, frames=weightDataStruct, interval=200, blit=True, repeat=True, repeat_delay=1000)
            HTML(ani.to_jshtml())
        else:
            plt.imshow(wgts_display_8bit)

    else:
        #if the arguement for slicing the x dimension is not none, then slice the plotted input
        if xSlice:
            plottedInput=input[timeframe,:,xSlice,:]
            #define a native figure size
            #nativeFigureSize=(plottedInput.shape[-2],plottedInput.shape[-3]*plottedInput.shape[-1]/dpiScale)
        else:
            plottedInput=input[timeframe,:,:,:]
            #define a native figure size
            #nativeFigureSize=(input.shape[-2]/dpiScale,input.shape[-3]*input.shape[-1]/dpiScale)

        nativeFigureSize=(plottedInput.shape[-2]/dpiScale,plottedInput.shape[-3]*plottedInput.shape[-1]/dpiScale)

        #if force resolution is defined then manually set it
        if forceResolution:
            if dpiScale !=30:
                print(f"dpiScale argument has been given a non-default value but forceResolution argument takes precedence. Fig size will be adjusted to {forceResolution}")
            #adjust the resolution to the forced one
            f, ax = plt.subplots(input.shape[-1],1,sharex=True,figsize=forceResolution)
        else:
            f, ax = plt.subplots(input.shape[-1],1,sharex=True,figsize=nativeFigureSize)
        
        #if the dpi parameter is defined
        if dpi:
            print(f"DPI value for figure changed to {dpi}, default matplotlib value is 100")
            #redefine the figure dpi
            f.dpi=dpi

        print(f'Figure size in pixels will be {f.get_size_inches()*f.dpi}')

        #print("DEBUG: Figure Created") #The figure can be created but not plotted
        #check if single image
        if input.shape[-1]==1:
            ax.imshow(plottedInput[...,0])
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            f.subplots_adjust(wspace=0, hspace=0)  #no improvement
            for i in range(input.shape[-1]):
                ax[i].imshow(plottedInput[...,i])
                ax[i].set_xticks([])
                ax[i].set_yticks([])
            #plt.subplots_adjust(wspace=0, hspace=0)

    #plt.show()
    if returnInput:
        return input
    else:
        return

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

def draw_edge_labels_no_overlap(
    G,
    pos,
    edge_labels=None,
    ax=None,
    label_pos=0.5,
    base_offset_px=6,
    repel_step_px=4,
    max_iter=40,
    pad_box=True,
    text_kw=None,
):
    """
    Draw edge labels for a NetworkX graph and automatically nudge them so they don't overlap.

    Parameters
    ----------
    G : nx.Graph / nx.DiGraph / nx.MultiGraph / nx.MultiDiGraph
    pos : dict node -> (x, y)
    edge_labels : dict with keys (u, v) or (u, v, k), values are strings
                  If None, uses edge attribute 'label' else 'weight' (if present).
    ax : matplotlib Axes
    label_pos : position along edge in [0, 1]
    base_offset_px : perpendicular offset (pixels) for parallel edges
    repel_step_px : pixel shove per iteration when labels overlap
    max_iter : max repel iterations
    pad_box : draw a white bbox behind text for legibility
    text_kw : kwargs forwarded to ax.text (e.g., fontsize=9)

    Returns
    -------
    dict: {(u, v[, k]): matplotlib.text.Text}
    """
    if ax is None:
        ax = plt.gca()
    text_kw = dict(text_kw or {})

    # --- helpers -------------------------------------------------------------
    def _unit_perp(u, v):
        x1, y1 = pos[u]; x2, y2 = pos[v]
        dx, dy = (x2 - x1), (y2 - y1)
        L = math.hypot(dx, dy)
        if L == 0:
            return (0.0, 0.0)
        return (-dy / L, dx / L)  # perpendicular

    def _edge_point(u, v, t):
        x1, y1 = pos[u]; x2, y2 = pos[v]
        return (x1*(1-t) + x2*t, y1*(1-t) + y2*t)

    def _pixels_to_data(dx_px, dy_px):
        inv = ax.transData.inverted()
        x0, y0 = ax.transData.transform((0, 0))
        x1, y1 = x0 + dx_px, y0 + dy_px
        (xd0, yd0) = inv.transform((x0, y0))
        (xd1, yd1) = inv.transform((x1, y1))
        return (xd1 - xd0, yd1 - yd0)

    def _group_parallel_edges():
        buckets = {}
        if G.is_multigraph():
            for u, v, k in G.edges(keys=True):
                buckets.setdefault(frozenset((u, v)), []).append((u, v, k))
        else:
            for u, v in G.edges():
                buckets.setdefault(frozenset((u, v)), []).append((u, v, None))
        for key in buckets:
            buckets[key].sort()
        return buckets

    def _iter_edges_with_keys():
        if G.is_multigraph():
            yield from G.edges(keys=True)
        else:
            for u, v in G.edges():
                yield (u, v, None)

    def _lookup_label(u, v, k):
        # user can pass (u, v), (v, u) for undirected, or (u, v, k)
        if edge_labels is None:
            d = G[u][v] if not G.is_multigraph() else G[u][v][k]
            return d.get("label", d.get("weight", None))
        # try the common keys
        if (u, v, k) in edge_labels: return edge_labels[(u, v, k)]
        if (u, v) in edge_labels:     return edge_labels[(u, v)]
        if not G.is_directed():
            if (v, u, k) in edge_labels: return edge_labels[(v, u, k)]
            if (v, u) in edge_labels:     return edge_labels[(v, u)]
        return None

    # --- place texts initially ----------------------------------------------
    parallel_groups = _group_parallel_edges()
    default_bbox = dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.2) if pad_box else None
    texts, positions = {}, {}

    for (u, v, k) in _iter_edges_with_keys():
        label = _lookup_label(u, v, k)
        if label is None or label == "":
            continue

        # midpoint + perpendicular offset for parallel edges
        x0, y0 = _edge_point(u, v, label_pos)
        nxp, nyp = _unit_perp(u, v)

        siblings = parallel_groups[frozenset((u, v))]
        idx = siblings.index((u, v, k))
        centered = idx - (len(siblings) - 1) / 2.0
        offx_data, offy_data = _pixels_to_data(base_offset_px * centered * nxp,
                                               base_offset_px * centered * nyp)
        px, py = x0 + offx_data, y0 + offy_data
        positions[(u, v, k)] = [px, py]

        t = ax.text(px, py, str(label), ha="center", va="center", bbox=default_bbox, **text_kw)
        texts[(u, v, k)] = t

    if not texts:
        return {}

    # render once to get bboxes
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def _bbox(edge_key):
        return texts[edge_key].get_window_extent(renderer=renderer).expanded(1.0, 1.05)

    # --- repel overlaps iteratively -----------------------------------------
    for _ in range(max_iter):
        moved = False
        for a, b in itertools.combinations(list(texts.keys()), 2):
            ba, bb = _bbox(a), _bbox(b)
            if not ba.overlaps(bb):
                continue

            # move apart along line connecting centers (in pixel space)
            ax_a = (ba.x0 + ba.x1) / 2.0; ay_a = (ba.y0 + ba.y1) / 2.0
            ax_b = (bb.x0 + bb.x1) / 2.0; ay_b = (bb.y0 + bb.y1) / 2.0
            vx, vy = (ax_a - ax_b), (ay_a - ay_b)
            dist = math.hypot(vx, vy)
            if dist < 1e-6:
                vx, vy, dist = 1.0, 0.0, 1.0
            ux, uy = vx / dist, vy / dist

            dx_data, dy_data = _pixels_to_data(repel_step_px * ux, repel_step_px * uy)
            positions[a][0] += dx_data; positions[a][1] += dy_data
            positions[b][0] -= dx_data; positions[b][1] -= dy_data
            texts[a].set_position(positions[a])
            texts[b].set_position(positions[b])
            moved = True

        if moved:
            fig.canvas.draw()
        else:
            break

    return texts

#draw a network diagram given a parameter path
#Author: Nick Bruns
#Created: 5/9/2025
#draw a network diagram given a parameter path
def drawNetwork(parameterFilePath,connectionLabelStyle='terse',includeOriginalConnectionName:bool=True,KKRedraw:bool=True,title:str=True,labelSegment:str=None,drawLegend:bool=True,saveToPath=None):

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

    #define constants for necessary values
    INHIBITORY_CHANNEL_CODE = 1
    EXCITATORY_CHANNEL_CODE = 0
    INACTIVE_CHANNEL_CODE = -1
    INHIBITORY_CHANNEL_CONNECTION_COLOR = 'red'
    EXCITATORY_CHANNEL_CONNECTION_COLOR = 'green'
    INACTIVE_CHANNEL_CONNECTION_COLOR = 'blue'
    NEUTRAL_CONNECTION_COLOR = 'black'

    MOMENTUM_CONNECTION_NAME='MomentumConn'
    IDENTITY_CONNECTION_NAME='IdentConn'
    CLONE_CONNECTION_NAME='CloneConn'
    TRANSPOSE_CONNECTION_NAME='TransposeConn'
    RESCALE_CONNECTION_NAME='RescaleConn'

    #clone type connections have a similar inherited type
    CLONE_TYPE_CONNECTION_SET=[CLONE_CONNECTION_NAME,TRANSPOSE_CONNECTION_NAME]
    IDENTITY_TYPE_CONNECTION_SET=[IDENTITY_CONNECTION_NAME,RESCALE_CONNECTION_NAME]

    IDENTITY_TYPE_LINE_STYLE='dotted'
    CLONE_TYPE_LINE_STYLE='solid'
    MOMENTUM_CONNECTION_LINE_STYLE='dashed'
    EDGE_LABEL_DEFAULT_COLOR='black'

    #definition of some various formating keywords
    DEFAULT_LINE_STYLE='solid'
    DEFAULT_NODE_COLOR='lightgreen'
    ARROW_SHAPE='-|>'
    SHORTHAND_CLONE_SYMBOL=r'$W_{Clone}$'#"W (clone)"
    SHORTHAND_MOMENTUM_SYMBOL="W"
    SHORTHAND_IDENT_SYMBOL=r'$A^{IdentConn}_{Identity}$'#"A"
    SHORTHAND_TRANSPOSE_SYMBOL=r'$W_{T}$'#"T"
    SHORTHAND_RESCALE_SYMBOL=r'$A^{RescaleConn}_{Identity}$'#"R"

    NETWORK_DIAGRAM_FILE_NAME='network.png'
    
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
    connectionChannelCode=[]
    for topLevelKey in parameters.keys():
        #use list comprehension to run through every key that should exist in a connection
        #then evaluate using 'in' to get a boolean for if the key is present
        #then use the all() to test if the boolean list is all true
        if all([key in parameters[topLevelKey] for key in connectionRequiredKeys]):
            connectionNames.append(topLevelKey)
            connectionType.append(parameters[topLevelKey]['groupType'])
            connectionPre.append(parameters[topLevelKey]['preLayerName'])
            connectionPost.append(parameters[topLevelKey]['postLayerName'])
            connectionChannelCode.append(parameters[topLevelKey]['channelCode'])
            #the use of get here preempts any situation where there is no originalConnName key
            tempOrigConnName=parameters[topLevelKey].get('originalConnName')
            #if the value is none then replace with noncharacter space
            if tempOrigConnName is None:
                tempOrigConnName = ''
            connectionOriginalConnName.append(tempOrigConnName)  
            print(f"{topLevelKey}\t\t{(parameters[topLevelKey]['preLayerName'])}\t{(parameters[topLevelKey]['postLayerName'])}")

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

    verboseConnectionEdgeLabels=dict()
    connectionStyleDict=dict()
    terseConnectionEdgeLabels=dict()
    copiedConnectionDict=dict()
    connectionColorDict=dict()

    #loop through connections, updating various style dictionaries as needed
    for name,pre,post,connType,origConnName,channelCode in zip(connectionNames,connectionPre,connectionPost,connectionType,connectionOriginalConnName,connectionChannelCode):
        #add to the connection label dictionary
        verboseConnectionEdgeLabels.update({(pre,post):f"{name}\n{connType}"})

        #update the dict for connections with duplicated outputs
        copiedConnectionDict.update({(pre,post):origConnName})

        #update style dictionary for setting connection properties
        if connType == MOMENTUM_CONNECTION_NAME:
            # a dict has to store this data to work
            connectionStyleDict.update({(pre,post):MOMENTUM_CONNECTION_LINE_STYLE})
        elif connType in CLONE_TYPE_CONNECTION_SET:
            connectionStyleDict.update({(pre,post):CLONE_TYPE_LINE_STYLE})
        elif connType in IDENTITY_TYPE_CONNECTION_SET:
            connectionStyleDict.update({(pre,post):IDENTITY_TYPE_LINE_STYLE})
        else:
            connectionStyleDict.update({(pre,post):DEFAULT_LINE_STYLE})
        
        #update dictionary of shorthand names
        if connType == MOMENTUM_CONNECTION_NAME:
            terseConnectionEdgeLabels.update({(pre,post):SHORTHAND_MOMENTUM_SYMBOL})
        elif connType == CLONE_CONNECTION_NAME:
            terseConnectionEdgeLabels.update({(pre,post):SHORTHAND_CLONE_SYMBOL})
        elif connType == IDENTITY_CONNECTION_NAME:
            terseConnectionEdgeLabels.update({(pre,post):SHORTHAND_IDENT_SYMBOL})
        elif connType == TRANSPOSE_CONNECTION_NAME:
            terseConnectionEdgeLabels.update({(pre,post):SHORTHAND_TRANSPOSE_SYMBOL})
        elif connType == RESCALE_CONNECTION_NAME:
            terseConnectionEdgeLabels.update({(pre,post):SHORTHAND_RESCALE_SYMBOL})
        else:
            terseConnectionEdgeLabels.update({(pre,post):''})

        #now update dictionaries of colors
        if channelCode == EXCITATORY_CHANNEL_CODE:
            connectionColorDict.update({(pre,post):EXCITATORY_CHANNEL_CONNECTION_COLOR})
        elif channelCode == INHIBITORY_CHANNEL_CODE:
            connectionColorDict.update({(pre,post):INHIBITORY_CHANNEL_CONNECTION_COLOR})
        elif channelCode == INACTIVE_CHANNEL_CODE:
            connectionColorDict.update({(pre,post):INACTIVE_CHANNEL_CONNECTION_COLOR})
        else:
            connectionColorDict.update({(pre,post):NEUTRAL_CONNECTION_COLOR})

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
        connectionLabelDisplay = verboseConnectionEdgeLabels
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

    fig,ax=plt.subplots(figsize=(10, 8))
    
    # Draw square-shaped nodes
    nx.draw_networkx_nodes(G, pos, node_color=DEFAULT_NODE_COLOR, node_size=1000, node_shape='s')

    # Draw labels on nodes
    if labelSegment:
        #create a new label dictionary with keys of the original name and values of the new parsed one
        newLabelDict=dict([(title,formatStringWithSeperators(title,insertionChar=labelSegment)) for title in verticalNodeOrder.keys()])

        #if label segment is not none, then add extra labels
        nx.draw_networkx_labels(G, pos, labels= newLabelDict, font_size=8)
    else:
        nx.draw_networkx_labels(G, pos, font_size=8)

    # Draw curved edges with arrows
    for i,edge in enumerate(G.edges()):
        nx.draw_networkx_edges(
            G, pos,
            style=connectionStyleDict[edge],    #drawing does not happen sequentially so a dict has to store everything
            edgelist=[edge],
            width=3,
            arrowstyle=ARROW_SHAPE,
            arrowsize=30,
            connectionstyle='arc3,rad=0.2',
            edge_color=connectionColorDict[edge]
        )
    # Draw the edge labels
    draw_edge_labels_no_overlap(G, pos, ax=ax,edge_labels=connectionLabelDisplay,label_pos=0.3,base_offset_px=1,max_iter=100,text_kw=dict(fontsize=8,color=EDGE_LABEL_DEFAULT_COLOR,verticalalignment='center'))

    if isinstance(title,str):
        plt.title(title)
    elif title==True:
        plt.title(parameterFilePath)
    plt.axis('off')

    if drawLegend is True:
        # Define colors and labels for legend
        color_map = {
            'Inhibitory Connection': INHIBITORY_CHANNEL_CONNECTION_COLOR,
            'Excitatory Connection': EXCITATORY_CHANNEL_CONNECTION_COLOR,
            'Inactive or Learning Connection': INACTIVE_CHANNEL_CONNECTION_COLOR,
        }

        custom_lines = [
            matplotlib.lines.Line2D([0], [0], color='gray', linestyle=CLONE_TYPE_LINE_STYLE, lw=2, label='Weight Clone Type Connections (TransposeConn or CloneConn)'),
            matplotlib.lines.Line2D([0], [0], color='gray', linestyle=MOMENTUM_CONNECTION_LINE_STYLE, lw=2, label='MomentumConn Connections (Learning)'),
            matplotlib.lines.Line2D([0], [0], color='gray', linestyle=IDENTITY_TYPE_LINE_STYLE, lw=2, label='Identity Type Connections (IdentConn or RescaleConn)')
        ]

        # Create custom legend handles
        legend_handles = [matplotlib.lines.Line2D([0], [0], marker='o', color='w', label=label,
                                markerfacecolor=color, markersize=10)
                        for label, color in color_map.items()]
        
        legend_handles.extend(custom_lines)

        plt.legend(handles=legend_handles, title="Legend")
    
    #if a none type, treat that as a request to save the figure to the folder where the python file is called
    if saveToPath is None:
        plt.savefig(NETWORK_DIAGRAM_FILE_NAME)
    #if a string, save there
    elif isinstance(saveToPath,str):
        plt.savefig(os.path.abspath(saveToPath))
    #if true, treat that as a resquest to save to the folder where the parameter file exists
    elif saveToPath is True:
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(parameterFilePath)),NETWORK_DIAGRAM_FILE_NAME))

    plt.show()

#to use commands from the command line, just type in the path to your python interpretter, the path of this .py file, the function name and then the arugments to give it
if __name__=="__main__":
    globals()[sys.argv[1]](sys.argv[2])