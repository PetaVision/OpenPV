import sys
import numpy
import pyopencl as cl
import numpy.linalg as la

## read the .cl file
#

if len(sys.argv) < 2:
    print("usage: test_build_cl filename")
    sys.exit(1)

fd = open(sys.argv[1], "r")
code = fd.read()

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

prg = cl.Program(ctx, code).build()

sys.exit(0)

