#!/usr/bin/python
import param_parse as pp
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('paramfile')
parser.add_argument('-l', help = 'path to directory with layers')
parser.add_argument('-c', help = 'path to directory with conns')
parser.add_argument('-p','--phase', help='layers colored by phase (default is scale)', action='store_true')
parser.add_argument('--legend', help='display simple legend on image', action='store_true')
args = parser.parse_args()

reader = pp.Param_Parser(args.paramfile, layers = args.l, conns = args.c)

if args.phase:
    if args.legend:
        pp.mermaid_writeout(reader.parse(), 'phase', True)
    else:
        pp.mermaid_writeout(reader.parse(), 'phase', False)
else:
    if args.legend:
        pp.mermaid_writeout(reader.parse(), 'scale', True)
    else:
        pp.mermaid_writeout(reader.parse(), 'scale', False)
