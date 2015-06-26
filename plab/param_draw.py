#!/usr/bin/python
import param_parse as pp
import sys
if len(sys.argv) > 1:
    filename = sys.argv[1]
    if len(sys.argv) > 3:
        reader = pp.Param_Parser(filename, layers = sys.argv[2], conns = sys.argv[3])
        pp.mermaid_writeout(reader.parse())
    else:
        print 'Warning: No layers/connections directory specified.'
        reader = pp.Param_Parser(filename, layers = '../src/layers', conns = '../src/connections/')
        pp.mermaid_writeout(reader.parse())
else:
    print 'No params file specified.'
