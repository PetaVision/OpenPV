#!/usr/bin/python
# Written by Max Theiler, max.theiler@gmail.com
# 2/5/2016

# This is the main param-parsing script for the draw
# tool. It performs two tasks: Reads/intrprets a PetaVision
# .params file, creates an internal representation using
# it's own objects, then writes out a .txt file which instructs
# the third-party drawing tool 'mermaid' on how to create an
# appropriate diagram. For mermaid input syntax, and other info, 
# go to https://github.com/knsv/mermaid

# Note that the firest couple object definitions (Col, Layer, Conn, and Param_Reader)
# were designed with modularity in mind. The idea was that other scripts might want
# to parse .params files for purposes other than diagramming, and these classes could be
# used for that.

# Also note that the mermaid writeout section of this file becomes much clearer if you can 
# look at its output. The bash 'draw' script which calls param_draw.py by default deletes the 
# output after the diagram is gereated, but you can retain the file with the -f flag. Manipulating
# this intermediate file directly can sometimes help to clean up a diagram if you don't want to
# delve into the python. To have mermaid generate from such a file, just enter "mermaid filename.txt"
# and check out the resulting "filename.txt.png" file.

import argparse
import re
import sys
import math
import os


class Col(object):
    def __init__(self):
        self.params = {}
    def __getitem__(self,item):
        return self.params[item]

class Layer(Col):
    def __init__(self, name, type, nodraw_status):
        self.type = type
        self.name = name
        self.params = {}
        self.color = None
        self.label = None
        self.nodraw = nodraw_status

class Conn(Col):
    def __init__(self, name, type, nodraw_status):
        self.type = type
        self.name = name
        self.params = {}
        self.color = None
        self.pre = None
        self.post = None
        self.label = None
        self.nodraw = nodraw_status

# Param reader converts reads a text params file and
# spawns objects to match those found. The main loop
# lies in the 'read()' method.
class Param_Reader(object):    
    # defining special regexes:
    comment_regex = re.compile('(.*)//')
    col_regex = re.compile('\s*HyPerCol\s*"\w+"\s*=\s*{')
    start_nodraw_regex = re.compile('^// START NODRAW')
    stop_nodraw_regex = re.compile('^// STOP NODRAW')
    nodraw_flag = False
    regex_dict = {
        'object_regex' : re.compile('(\w+)\s*"(\w+)"\s*=\s*{'),
        'end_regex' : re.compile('};'),
        'param_regex' : re.compile('@?(\w+)\s*=\s*"?(-?[^"]+)"?;'),
        'include_regex' : re.compile('#include\s+"(\w+)"'),
        }
    # sort categorized objects into respective master lists of Layers and Conns
    def assign_object(self,type,name):
        official_layers,official_conns = self.lists
        if type in official_layers:
            self.layer_dict[name] = Layer(name,type,self.nodraw_flag)
            self.current_object = self.layer_dict[name]
            self.layers_in_order.append(name)
            return
        elif type in official_conns:
            self.conn_dict[name] = Conn(name,type,self.nodraw_flag)
            self.current_object = self.conn_dict[name]
            self.conns_in_order.append(name)
            return
        if not official_layers or not type in official_layers:
            if re.search('Layer',type) or re.search('Movie',type):
                self.layer_dict[name] = Layer(name,type,self.nodraw_flag)
                self.current_object = self.layer_dict[name]
                self.layers_in_order.append(name)
                if official_layers:
                    print('Warning: Layer type ' + type + ' not named in layers directory')
                return
        if not official_conns or not type in official_conns:
            if re.search('Conn',type):
                self.conn_dict[name] = Conn(name,type,self.nodraw_flag)
                self.current_object = self.conn_dict[name]
                self.conns_in_order.append(name)
                if official_conns:
                    print('Warning: Conn type ' + type + ' not named in conns directory')
                return
    # method to determine object type from regex matches 
    def interpret(self, pattern, match):
        if pattern is 'object_regex':
            name = match.group(2)
            type = match.group(1)
            self.assign_object(type,name)
        elif pattern is 'end_regex':
            self.current_object = None
        elif pattern is 'param_regex' and self.current_object:
            label = match.group(1)
            value = match.group(2)
            self.current_object.params[label] = value
        elif pattern is 'include_regex':
            parent = match.group(1)
            if parent in self.layer_dict:
                self.current_object.params = self.layer_dict[parent].params.copy()
            elif parent in self.conn_dict:
                self.current_object.params = self.conn_dict[parent].params.copy()
    # setting the NODRAW flag
    def check_nodraw(self,line):
        if not self.nodraw_flag:
            match = re.search(self.start_nodraw_regex, line)
            if match:
                self.nodraw_flag = True
        elif self.nodraw_flag:
            match = re.search(self.stop_nodraw_regex, line)
            if match:
                self.nodraw_flag = False
    # main loop that reads param file. loop thru each
    # line, determine via regex match what we're looking at
    def read(self):
        with open(self.filename, "r") as file:
            for line in file:
                self.check_nodraw(line)
                match = re.search(self.comment_regex, line)
                if match:
                    line = match.group(1)
                match = re.search(self.col_regex, line)
                if match:
                    self.current_object = self.column
                for regex_pattern in self.regex_dict:
                    match = re.search(self.regex_dict[regex_pattern], line)
                    if match:
                        self.interpret(regex_pattern, match)
                        break

    def __init__(self, filename, lists):
        self.filename = filename
        self.lists = lists
        self.layer_dict = {}
        self.layers_in_order = []
        self.conn_dict = {}
        self.conns_in_order = []
        self.column = Col()
        self.current_object = None
        self.read()

# Param parser categorizes the objects read in param reader and sets up
# the Conn-relationships between layers, mostly via parameters form the file. 
class Param_Parser(Param_Reader):
    # check if the layer/conn type is actually listed in PetaVision's
    # src directory.
    def official_lists(self,layer_dir,conn_dir):
        official_layers = []
        official_conns = []
        if os.path.isdir(layer_dir) and os.path.isdir(conn_dir):
            lay = os.listdir(layer_dir)
            con = os.listdir(conn_dir)
            for i in lay:
                match = re.search('(.+).cpp',i)
                if match:
                    official_layers.append(match.group(1))
            for i in con:
                match = re.search('(.+).cpp',i)
                if match:
                    official_conns.append(match.group(1))
        if not os.path.isdir(layer_dir):
            print('Warning: Directory "' + layer_dir + '" not found, attempting to infer Layers.')
        if not os.path.isdir(conn_dir):
            print('Warning: Directory "' + conn_dir + '" not found, attempting to infer Conns')
        return official_layers, official_conns
    # get scale numerically, in case color-by-scale is needed
    def calc_scale(self):
        cnx = float(self.column['nx'])
        cny = float(self.column['ny'])
        for i in self.layer_dict.values():
            if 'nxScale' in i.params and 'nyScale' in i.params:
                i.params['nx'] = float(i['nxScale'])*cnx
                i.params['ny'] = float(i['nyScale'])*cny
    # used if the Conn doesn't have a preLayerName. it attempts to find one by
    # exploiting the (informal) Conn naming conventions
    def infer_pre_from_name(self,conn_name):
        found = False
        for i in self.layer_dict.values():
            pre_regex = ('^' + i.name)
            if re.search(pre_regex,conn_name):
                print(i.name + ' inferred as preLayer of ' + conn_name)
                return i.name
        print('Could not infer prelayer of ' + conn_name)
    # see above
    def infer_post_from_name(self,conn_name):
        found = False
        for i in self.layer_dict.values():
            post_regex = (i.name + '$')
            if re.search(post_regex,conn_name):
                print(i.name + ' inferred as postLayer of ' + conn_name)
                return i.name
        print('Could not infer postlayer of ' + conn_name)
    # used to set up each conn object so that it understands what its pre and post layers are
    def relate_objects(self):
        remove_list = []
        for i in self.conn_dict.values():
            if 'preLayerName' in i.params:
                if not i['preLayerName'] in self.layer_dict:
                    print('Warning: ' + i['preLayerName'] + ' in ' + i.name + ' not found in param layers')
                    continue
                i.pre = i['preLayerName']
            else:
                print('Warning: ' + i.name + ' does not specify a preLayer')
                i.pre = self.infer_pre_from_name(i.name)

            if 'postLayerName' in i.params:
                if not i['postLayerName'] in self.layer_dict:
                    print('Warning: ' + i['postLayerName'] + ' in ' + i.name + ' not found in param layers')
                    continue
                i.post = i['postLayerName']
            else:
                print('Warning: ' + i.name + ' does not specify a postLayer')
                i.post = self.infer_post_from_name(i.name)

            if not i.pre and not i.post:
                remove_list.append(i.name)
                
        for i in remove_list:
            print('Deleting ' + i)
            del self.conn_dict[i]
            del self.conns_in_order[self.conns_in_order.index(i)]
    # if conns are clones of eachother, recurse back through the
    # parent of each clone until we hit an original parent. that
    # parent is assigned a label (labels are the integers used to 
    # group conns in the final display)
    def original_conn_label(self,conn):
        parent = self.conn_dict[conn.params['originalConnName']]
        if parent.label:
            return parent.label
        elif 'originalConnName' in parent.params:
            return self.original_conn_label(parent)
        else:
            self.cl = self.cl+1
            parent.label = str(self.cl)
            return str(self.cl)
    # Loop through the conn list and assign labels from the parent labels.
    # also assign 'MAX' and 'SUM' labels for pooling conns
    def assign_labels(self):
        self.cl = 0
        for i in self.conns_in_order:
            conn = self.conn_dict[i]
            if 'originalConnName' in conn.params:
                if conn['originalConnName'] in self.conn_dict:
                    originalConnName = conn['originalConnName']
                    originalConn = self.conn_dict[originalConnName]
                    conn.label = self.original_conn_label(conn)
                    if conn.type == 'TransposePoolingConn':
                        pool_type = originalConn['pvpatchAccumulateType']
                        if pool_type == 'maxpooling':
                            conn.label = ('MAX_' + conn.label)
                        if pool_type == 'sumpooling':
                            conn.label = ('SUM_' + conn.label)
                else:
                    print('Warning: originalConn "' + conn['originalConnName'] + '" of ' + conn.name + ' not found.') 
            
        for i in self.conn_dict.values():
            if i.type == 'PoolingConn':
                pool_type = i['pvpatchAccumulateType']
                if pool_type == 'maxpooling':
                    if i.label:
                        i.label = ('MAX_' + i.label)
                    else:
                        i.label = ('MAX')
                elif pool_type == 'sumpooling':
                    if i.label:
                        i.label = ('SUM_' + i.label)
                    else:
                        i.label = ('SUM')

    # For layers which copy other layers via an originalLayerName 
    # parameter and are not technically connected, create an
    # artifical internal conn for drawing the connection 
    def make_original_layer_conns(self):
        for i in self.layer_dict.values():
            if 'originalLayerName' in i.params:
                c = Conn(i.name,'OriginalLayerCopy',i.nodraw)
                c.post = i.name
                c.pre = i['originalLayerName']
                self.conn_dict[c.name] = c
    # see above
    def make_pooling_layer_conns(self):
        for i in self.conn_dict.values():
            if 'postIndexLayerName' in i.params:
                c = Conn((i['postIndexLayerName'] + '_conn'), 'IndexLayerCopy',i.nodraw)
                c.post =  i.post
                c.pre = i['postIndexLayerName']
                self.conn_dict[c.name] = c

    # take all of the layers/conns that were flagged NODRAW
    # out of the master lists
    def remove_nodraws(self):
        remove_list = []
        for l in self.layer_dict.values():
            if l.nodraw:
                remove_list.append(l.name)
                for c in self.conn_dict.values():
                    if c.pre == l.name or c.post == l.name:
                        remove_list.append(c.name)
        for c in self.conn_dict.values():
            if c.nodraw:
                remove_list.append(c.name)
        remove_list = set(remove_list)
        for i in remove_list:
            if i in self.layer_dict:
                del self.layer_dict[i]
                del self.layers_in_order[self.layers_in_order.index(i)]
            if i in self.conn_dict:
                del self.conn_dict[i]
                del self.conns_in_order[self.conns_in_order.index(i)]
    # parse basically just runs through the object's methods. 
    # this kept separate from __init__ in hopes that the class
    # and its methods could be used independently by another
    # script someday, or used interactively.
    def parse(self):
        self.relate_objects()
        self.assign_labels()
        self.calc_scale()
        self.make_original_layer_conns()
        self.make_pooling_layer_conns()
        self.remove_nodraws()
        return [self.layer_dict,self.conn_dict,self.layers_in_order,self.conns_in_order]

    def __init__(self, filename, **kwargs):
        laydir = kwargs.get('layers','')
        condir = kwargs.get('conns','')
        self.filename = filename
        self.lists = self.official_lists(laydir,condir)
        self.layer_dict = {}
        self.layers_in_order = []
        self.conn_dict = {}
        self.conns_in_order = []
        self.column = Col()
        self.current_object = None
        self.read()
        if kwargs['alph']:
            self.layers_in_order = sorted(self.layers_in_order)
            self.conns_in_order = sorted(self.conns_in_order)

# This method for printing out the text file which mermaid uses to construct its diagram.
# mermaid uses a lot of HTML-style code for line width and color and such.

# This section becomes much clearer if you can look at its output. The bash 'draw' script
# which calls param_draw.py by default deletes the output after the diagram is gereated,
# but you can retain the file with the -f flag.
def mermaid_writeout(parser_output, colorby, legend):
    layer_dict = parser_output[0]
    conn_dict = parser_output[1]
    layers_in_order = parser_output[2]
    conns_in_order = parser_output[3]

    # Assigning which types of conns are going to get which colors/styles, and what those styles are
    # according to the format (conn type), (style definition) = 'a string', 'a string'
    dash_type,dash_text = ['TransposeConn','TransposePoolingConn'],',stroke-dasharray: 10, 10'
    dot_type,dot_text = 'IdentConn',',stroke-dasharray: 2, 2'
    blue_type,blue_code = ['IndexLayerCopy','OriginalLayerCopy'],',stroke:#00f'
    thick_param,thick_val,thick = 'plasticityFlag','true',',stroke-width:4px'
    green_param,green_val,green_code = 'channelCode','0',',stroke:#0f0'
    red_param,red_val,red_code = 'channelCode','1',',stroke:#f00'
    
    # set color values to be proportionate to the scale of the layer in the Col,
    # if user has specified that color scheme
    def calculate_scale_colorvalues():
        mincolor = [0x99,0x99,0xdd]
        maxcolor = [0xee,0xee,0xff]
        scales = {}
        for i in layer_dict.values():
            if 'nx' in i.params and 'ny' in i.params:
                s = i['nx']*i['ny']
                if s in scales:
                    scales[s].append(i.name)
                else:
                    scales[s] = []
                    scales[s].append(i.name)
        ordered_scales = sorted(scales)
        myc = mincolor
        for i in ordered_scales:
            myc_z = zip(maxcolor,mincolor,myc)
            myc = [myc + (max-min)/len(ordered_scales) for max,min,myc in myc_z]
            for j in scales[i]:
                strcolor = [str(hex(int(c)))[2:4] for c in myc]
                strcolor = ''.join(strcolor)
                layer_dict[j].color = strcolor
        
    # set color values to be proportionate to the phase of the layer,
    # if user has specified that color scheme
    def calculate_phase_colorvalues():
        mincolor = [0x99,0x99,0xdd]
        maxcolor = [0xee,0xee,0xff]
        phases = {}
        for i in layer_dict.values():
            if 'phase' in i.params:
                p = int(i['phase'])
                if p in phases:
                    phases[p].append(i.name)
                else:
                    phases[p] = []
                    phases[p].append(i.name)
        ordered_phases = sorted(phases)
        myc = mincolor
        for i in ordered_phases:
            myc_z = zip(maxcolor,mincolor,myc)
            myc = [myc + (max-min)/len(ordered_phases) for max,min,myc in myc_z]
            for j in phases[i]:
                strcolor = [str(hex(int(c)))[2:4] for c in myc]
                strcolor = ''.join(strcolor)
                layer_dict[j].color = strcolor
    # Certain starts of object names are reserved by mermaid: this method just
    # appends junk characters for the known reserved terms if it sees them to
    # avoid them being parsed by mermaid. 
    def append_dots():
        regex_list = ['v$',
                      'style$'
                      'default$',
                      'linkStyle$'
                      'classDef$'
                      'class$'
                      'click$'
                      'graph$'
                      'subgraph$'
                      'end$']
        for r in regex_list:
            for i in layer_dict.values():
                if re.search(r,i.name):
                    print('Warning: ' + i.name + ' ends in a mermaid-reserved string. Appending junk characters to input file.')
                    i.name = i.name + 'xxx'
            for i in conn_dict.values():
                if re.search(r,i.pre):
                    i.pre = i.pre + 'xxx'
                if re.search(r,i.post):
                    i.post = i.post + 'xxx'
    # end append_dots()

    # Main writeout section of mermaid writeout method:
    if colorby == 'phase':
        calculate_phase_colorvalues()
    elif colorby == 'scale':
        calculate_scale_colorvalues()
    else:
        calculate_scale_colorvalues()

    f = open('mermaid_input', 'w')
    f.write('graph BT;\n')
    append_dots()

    if legend:
        f.write('leg[Legend:<br>'
                'Layer Shade = ' + colorby + '<br>'
                'Diamond     = HyperLCALayer<br>'   
                'Green  = Excitatory Conn<br>'
                'Red    = Inhibatory Conn<br>'
                'Dashed = TransposeConn<br>'
                'Dotted = IdentConn<br>'
                'Thick  = Plastic Weights<br>'
                'Blue   = LayerCopy<br>'
                'Numbers represent groups<br>'
                'of duplicated/transposed<br>'
                'connections.];')
    # Loop thru layers, printout a node for each. Use {layername} 
    # syntax to draw diamonds for HyPerLCAs, [layername] for squares, 
    # i.e. everything else.
    for i in layers_in_order:
        label = layer_dict[i].name
        if re.search('xxx$',layer_dict[i].name):
            label = layer_dict[i].name[0:-4]
        if layer_dict[i].type == 'HyPerLCALayer':
            f.write(layer_dict[i].name + '{' + label + '};\n')
        else: 
            f.write(layer_dict[i].name + '[' + label + '];\n')

    # Loop thru conns, print out an edge for each. Mermaid's
    # edge syntax is, prelayer-->postlayer; for unlabeled edges 
    # and prelayer-->|text|postlayer; for labeled edges
    for i in conn_dict.values():
        if i.pre and i.post:
            f.write(i.pre + "-->")
            if i.label:
                f.write("|" + i.label + "|")
            f.write(i.post + ";\n")

    # In this script, a "classDef" of style is defined separately for each node
    # and immediately applied by "class" to the node in question. classDef is
    # meant to be a modular definition, but for simplicity it is recreated here
    # for each node.
    for i in layer_dict.values():
        n = i.name
        if not i.color:
            i.color = 'aaeeee'
        f.write('classDef color' + n +' fill:' + i.color + ';\n')
        f.write('class ' + n + ' color' + n + ";\n")

    # Styling edges is done with "linkStyle." It's a bit trickier, 
    # and order matters, but the same principle of a redundantly 
    # making a new style definition for every edge is applied.
    link_name_iter = 0
    for i in conn_dict.values():
        color = ',stroke:#000000'
        size = ',stroke-width:2px'
        dasharray = ''
        
        if thick_param in i.params:
            if i[thick_param] == thick_val:
                size = thick

        if i.type in dash_type:
            dasharray = dash_text
        elif i.type == dot_type:
            dasharray = dot_text
        if i.type in blue_type:
            color = blue_code

        if green_param in i.params:
            if i[green_param] == green_val:
                color = green_code
        if red_param in i.params:
            if i[red_param] == red_val:
                color = red_code
                
        if i.pre and i.post:
            f.write('linkStyle ' + str(link_name_iter) + ' fill:none' + color + size + dasharray + ';\n')
            link_name_iter = link_name_iter+1

# Function call -- when the bash script is run or if the
# python script itself is run.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('paramfile')
    parser.add_argument('-l', help = 'path to directory with layers')
    parser.add_argument('-c', help = 'path to directory with conns')
    parser.add_argument('-p','--phase', help='layers colored by phase (default is scale)', action='store_true')
    parser.add_argument('--alphabet', help='order params file layers alphabetically', action='store_true')
    parser.add_argument('--legend', help='display simple legend on image', action='store_true')
    args = parser.parse_args()

    reader = Param_Parser(args.paramfile, layers = args.l, conns = args.c, alph = args.alphabet)

    if args.phase:
            mermaid_writeout(reader.parse(), 'phase', args.legend)
    else:
            mermaid_writeout(reader.parse(), 'scale', args.legend)
