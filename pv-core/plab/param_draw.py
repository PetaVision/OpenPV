#!/usr/bin/python

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
    def __init__(self, name, type):
        self.type = type
        self.name = name
        self.params = {}
        self.color = None
        self.label = None

class Conn(Col):
    def __init__(self, name, type):
        self.type = type
        self.name = name
        self.params = {}
        self.color = None
        self.pre = None
        self.post = None
        self.label = None

class Param_Reader(object):    
    comment_regex = re.compile('(.*)//')
    col_regex = re.compile('\s*HyPerCol\s*"\w+"\s*=\s*{')
    regex_dict = {
        'object_regex' : re.compile('(\w+)\s*"(\w+)"\s*=\s*{'),
        'end_regex' : re.compile('};'),
        'param_regex' : re.compile('@?(\w+)\s*=\s*"?(-?[^"]+)"?;'),
        'include_regex' : re.compile('#include\s+"(\w+)"'),
        }
    def assign_object(self,type,name):
        official_layers,official_conns = self.lists
        if type in official_layers:
            self.layer_dict[name] = Layer(name,type)
            self.current_object = self.layer_dict[name]
            self.layers_in_order.append(name)
            return
        elif type in official_conns:
            self.conn_dict[name] = Conn(name,type)
            self.current_object = self.conn_dict[name]
            self.conns_in_order.append(name)
            return
        if not official_layers or not type in official_layers:
            if re.search('Layer',type) or re.search('Movie',type):
                self.layer_dict[name] = Layer(name,type)
                self.current_object = self.layer_dict[name]
                self.layers_in_order.append(name)
                if official_layers:
                    print('Warning: Layer type ' + type + ' not named in layers directory')
                return
        if not official_conns or not type in official_conns:
            if re.search('Conn',type):
                self.conn_dict[name] = Conn(name,type)
                self.current_object = self.conn_dict[name]
                self.conns_in_order.append(name)
                if official_conns:
                    print('Warning: Conn type ' + type + ' not named in conns directory')
                return
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

    def read(self):
        with open(self.filename, "r") as file:
            for line in file:
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

class Param_Parser(Param_Reader):

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

    def calc_scale(self):
        cnx = float(self.column['nx'])
        cny = float(self.column['ny'])
        for i in self.layer_dict.values():
            if 'nxScale' in i.params and 'nyScale' in i.params:
                i.params['nx'] = float(i['nxScale'])*cnx
                i.params['ny'] = float(i['nyScale'])*cny

    def infer_pre_from_name(self,conn_name):
        found = False
        for i in self.layer_dict.values():
            pre_regex = ('^' + i.name)
            if re.search(pre_regex,conn_name):
                print(i.name + ' inferred as preLayer of ' + conn_name)
                return i.name
        print('Could not infer prelayer of ' + conn_name)

    def infer_post_from_name(self,conn_name):
        found = False
        for i in self.layer_dict.values():
            post_regex = (i.name + '$')
            if re.search(post_regex,conn_name):
                print(i.name + ' inferred as postLayer of ' + conn_name)
                return i.name
        print('Could not infer postlayer of ' + conn_name)
            
    def relate_objects(self):
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

    def assign_labels(self):
        self.cl = 0
        for i in self.conns_in_order:
            conn = self.conn_dict[i]
            if 'originalConnName' in conn.params:
                if conn['originalConnName'] in self.conn_dict:
                    conn.label = self.original_conn_label(conn)
                    if conn.type == 'TransposePoolingConn':
                        pool_type = conn['pvpatchAccumulateType']
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
                    i.label = ('MAX_' + i.label)
                elif pool_type == 'sumpooling':
                    i.label = ('SUM_' + i.label)

    def make_original_layer_conns(self):
        for i in self.layer_dict.values():
            if 'originalLayerName' in i.params:
                c = Conn(i.name,'OriginalLayerCopy')
                c.post = i.name
                c.pre = i['originalLayerName']
                self.conn_dict[c.name] = c

    def make_pooling_layer_conns(self):
        for i in self.conn_dict.values():
            if 'postIndexLayerName' in i.params:
                c = Conn((i['postIndexLayerName'] + '_conn'), 'IndexLayerCopy')
                c.post =  i.post
                c.pre = i['postIndexLayerName']
                self.conn_dict[c.name] = c

    def parse(self):
        self.relate_objects()
        self.assign_labels()
        self.calc_scale()
        self.make_original_layer_conns()
        self.make_pooling_layer_conns()
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

def mermaid_writeout(parser_output, colorby, legend):
    layer_dict = parser_output[0]
    conn_dict = parser_output[1]
    layers_in_order = parser_output[2]
    conns_in_order = parser_output[3]
    dash_type,dash_text = ['TransposeConn','TransposePoolingConn'],',stroke-dasharray: 10, 10'
    dot_type,dot_text = 'IdentConn',',stroke-dasharray: 2, 2'
    blue_type,blue_code = ['IndexLayerCopy','OriginalLayerCopy'],',stroke:#00f'
    thick_param,thick_val,thick = 'plasticityFlag','true',',stroke-width:4px'
    green_param,green_val,green_code = 'channelCode','0',',stroke:#0f0'
    red_param,red_val,red_code = 'channelCode','1',',stroke:#f00'
    
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
    for i in layers_in_order:
        label = layer_dict[i].name
        if re.search('xxx$',layer_dict[i].name):
            label = layer_dict[i].name[0:-4]
        if layer_dict[i].type == 'HyPerLCALayer':
            f.write(layer_dict[i].name + '{' + label + '};\n')
        else: 
            f.write(layer_dict[i].name + '[' + label + '];\n')

    for i in conn_dict.values():
        if i.pre and i.post:
            f.write(i.pre + "-->")
            if i.label:
                f.write("|" + i.label + "|")
            f.write(i.post + ";\n")

    for i in layer_dict.values():
        n = i.name
        if not i.color:
            i.color = 'aaeeee'
        f.write('classDef color' + n +' fill:' + i.color + ';\n')
        f.write('class ' + n + ' color' + n + ";\n")
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

# Function call

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('paramfile')
    parser.add_argument('-l', help = 'path to directory with layers')
    parser.add_argument('-c', help = 'path to directory with conns')
    parser.add_argument('-p','--phase', help='layers colored by phase (default is scale)', action='store_true')
    parser.add_argument('--legend', help='display simple legend on image', action='store_true')
    args = parser.parse_args()

    reader = Param_Parser(args.paramfile, layers = args.l, conns = args.c)

    if args.phase:
        if args.legend:
            mermaid_writeout(reader.parse(), 'phase', True)
        else:
            mermaid_writeout(reader.parse(), 'phase', False)
    else:
        if args.legend:
            mermaid_writeout(reader.parse(), 'scale', True)
        else:
            mermaid_writeout(reader.parse(), 'scale', False)
