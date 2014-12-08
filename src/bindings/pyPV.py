from ctypes import *
import sys
lib = cdll.LoadLibrary('../../lib/libpv.so')
mpi =  CDLL('libmpi.so', RTLD_GLOBAL)

class pyHyPerCol(object):
   def __init__(self, argc, argv):
      self.obj = lib.pvBuild(argc, argv)

   def run(self):
      return lib.pvRun(self.obj)


#Test script
if __name__ == "__main__":
   c_argc = 4
   c_argv = (c_char_p * c_argc)("pv", "-p", "/home/slundquist/workspace/PVSystemTests/BasicSystemTest/input/BasicSystemTestAbsolute.params", "-t")
   pvObj = pyHyPerCol(c_argc, c_argv)
   pvObj.run()
