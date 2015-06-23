
#!/usr/bin/python

import re
import sys
import math
import os

def official_lists(layer_dir = '/layers', conn_dir = '/connections'):
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
            re.search('(.+).cpp',i)
            if match:
                official_conns.append(match.group(1))
    if not os.path.isdir(layer_dir):
        print('Warning: Directory "' + layer_dir + '" not found, attempting to infer Layers.')
    if not os.path.isdir(conn_dir):
        print('Warning: Directory "' + conn_dir + '" not found, attempting to infer Conns')
    return official_layers, official_conns

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
    comment_regex = re.compile('(.*)//') # matches everything before a comment, including ''
    col_regex = re.compile('\s*HyPerCol\s*"\w+"\s*=\s*{')
    regex_dict = {
        'object_regex' : re.compile('(\w+)\s*"(\w+)"\s*=\s*{'), # grp 1 type, grp 2 name
        'end_regex' : re.compile('};'),
        'param_regex' : re.compile('@?(\w+)\s*=\s*"?(-?[^"]+)"?;'), # grp 1 param, grp 2 val
        'include_regex' : re.compile('#include\s+"(\w+)"'),}

    def assign_object(self,type,name):
        official_layers,official_conns = self.lists
        if type in official_layers:
            self.layer_dict[name] = Layer(name,type)
            self.current_object = self.layer_dict[name]
            self.layers_in_order.append(name)
        elif type in official_conns:
            self.conn_dict[name] = Conn(name,type)
            self.current_object = self.conn_dict[name]
            self.conns_in_order.append(name)
        if not official_layers:
            if re.search('Layer',type) or re.search('Movie',type):
                self.layer_dict[name] = Layer(name,type)
                self.current_object = self.layer_dict[name]
                self.layers_in_order.append(name)
        if not official_conns:
            if re.search('Conn',type):
                self.conn_dict[name] = Conn(name,type)
                self.current_object = self.conn_dict[name]
                self.conns_in_order.append(name)

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
            elif name in self.conn_dict:
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
    def calc_scale(self):
        cnx = float(self.column['nx'])
        cny = float(self.column['ny'])
        for i in self.layer_dict.itervalues():
            if 'nxScale' in i.params and 'nyScale' in i.params:
                i.params['nx'] = float(i['nxScale'])*cnx
                i.params['ny'] = float(i['nyScale'])*cny

    def relate_objects(self):
        for i in self.conn_dict.itervalues():
            if 'preLayerName' in i.params:
                #i.pre = i['preLayerName']
                if not i['preLayerName'] in self.layer_dict:
                    print 'Warning: ' + i['preLayerName'] + ' in ' + i.name + ' not found in Layers'
                    continue
                i.pre = i['preLayerName']
            if 'postLayerName' in i.params:
                #i.post = i['postLayerName']
                if not i['postLayerName'] in self.layer_dict:
                    print 'Warning: ' + i['postLayerName'] + ' in ' + i.name + ' not found in Layers'
                    continue
                i.post = i['postLayerName']


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
                conn.label = self.original_conn_label(conn)

    def make_original_layer_conns(self):
        for i in self.layer_dict.itervalues():
            if 'originalLayerName' in i.params:
                c = Conn(i.name,'LayerCopy')
                c.post = i.name
                c.pre = i['originalLayerName']
                self.conn_dict[c.name] = c

def mermaid_writeout(layer_dict,conn_dict,layers_in_order,conns_in_order):
    dash_type,dash_text = 'TransposeConn',',stroke-dasharray: 10, 10'
    dot_type,dot_text = 'IdentConn',',stroke-dasharray: 2, 2'
    blue_type,blue_code = 'LayerCopy',',stroke:#00f'
    thick_param,thick_val,thick = 'plasticityFlag','true',',stroke-width:4px'
    green_param,green_val,green_code = 'channelCode','0',',stroke:#0f0'
    red_param,red_val,red_code = 'channelCode','1',',stroke:#f00'
    
    def calculate_scale_colorvalues():
        mincolor = [0x99,0x99,0xdd]
        maxcolor = [0xee,0xee,0xff]
        scales = {}
        for i in layer_dict.itervalues():
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
                strcolor = [str(hex(c))[2:4] for c in myc]
                strcolor = ''.join(strcolor)
                layer_dict[j].color = strcolor
        
    calculate_scale_colorvalues()

    f = open('param_graph', 'w')
    f.write('graph BT;\n')
    for i in layers_in_order:
        if layer_dict[i].type == 'HyPerLCALayer':
            f.write(i + '{' + i + '};\n')
        else: 
            f.write(i + '[' + i + '];\n')

    for i in conn_dict.itervalues():
        if i.pre and i.post:
            f.write(i.pre + "-->")
            if i.label:
                f.write("|" + i.label + "|")
            f.write(i.post + ";\n")

    for i in layer_dict.itervalues():
        n = i.name
        if not i.color:
            i.color = 'aaeeee'
        f.write('classDef color' + n +' fill:' + i.color + ';\n')
        f.write('class ' + n + ' color' + n + ";\n")
    link_name_iter = 0
    for i in conn_dict.itervalues():
        color = ',stroke:#000000'
        size = ',stroke-width:2px'
        dasharray = ''
        
        if thick_param in i.params:
            if i[thick_param] == thick_val:
                size = thick

        if i.type == dash_type:
            dasharray = dash_text
        elif i.type == dot_type:
            dasharray = dot_text
        if i.type == blue_type:
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




#HAVE TO SET FLAG CHECKING TO BE MORE ROBUST
param_name = sys.argv[1]
if len(sys.argv) == 4:
    off_lists = official_lists(sys.argv[2],sys.argv[3])
else:
    off_lists = official_lists()
if len(sys.argv) < 5:
    pd = Param_Parser(param_name, off_lists)
    pd.relate_objects()
    pd.assign_labels()
    pd.calc_scale()
    pd.make_original_layer_conns()
    mermaid_writeout(pd.layer_dict,pd.conn_dict,pd.layers_in_order,pd.conns_in_order)
if len(sys.argv) == 5 and sys.argv[4] == '-a':
    # Do the operations for the analysis script
    pass
