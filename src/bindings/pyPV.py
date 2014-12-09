from ctypes import *
import sys

class pyHyPerCol(object):
   #Arguments is a list of parameter strings
   def __init__(self, arguments):
      #If use linux
      if sys.platform == "linux" or sys.platform == "linux2":
         self.lib = cdll.LoadLibrary('../../lib/libpv.so')
         mpi =  CDLL('libmpi.so', RTLD_GLOBAL)
      elif sys.platform == "darwin":
         #If use mac
         self.lib = cdll.LoadLibrary('../../lib/libpv.dylib')
         self.lib.pvBuild.restype = POINTER(c_char_p)
         mpi =  CDLL('libmpi.dylib', RTLD_GLOBAL)
      elif sys.platform == "win32":
         print "Windows not supported"
         sys.exit()
      else:
         print "Operating system", sys.platform, "not known"
         sys.exit()
      argc = len(arguments)
      argv = (c_char_p * (argc+1))()
      for i in range(argc):
         argv[i] = arguments[i]
      #Last element in argv should be NULL
      argv[argc] = None

      self.hc = self.lib.pvBuild(argc, argv)

   def run(self):
      return self.lib.pvRun(self.hc)


#Test scripti
if __name__ == "__main__":
   #c_argc = 4
   #c_argv = (c_char_p * (c_argc+1))("pv", "-p", "input/BasicSystemTest.params", "-t", None)
   args = ["pv", "-p", "input/BasicSystemTest.params", "-t"]
   pvObj = pyHyPerCol(args)
   pvObj.run()
