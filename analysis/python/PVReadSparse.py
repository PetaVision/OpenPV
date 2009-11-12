from numpy import *

class PVReadSparse(object):
   """A class to read PetaVision sparse activity files"""

   def __init__(self, filename):
      """Constructor: Open file and read in file metadata into params"""
      
      self.open(filename)
      self.read_params()
      self.time = 0
      self.timestep = 0

   # end __init__

   def open(self, filename):
      """Open given filename"""
      self.filename = filename
      self.file = open(filename, mode='rb')
   # end open
   
   def next_record(self):
      """Read the next record and return an ndarray"""
      self.time = fromfile(self.file, 'd', 1) [0]
      numItems = fromfile(self.file, 'i', 1) [0]
      if (numItems > 0):
         activity = fromfile(self.file, 'i', numItems)
      else:
         activity = []
      self.timestep += 1
      return activity
   # end next

   def read_params(self):
      """Read the file metadata parameters"""
      filepos = self.file.tell()
      self.numParams = fromfile(self.file, 'i', 2) [1]
      self.file.seek(filepos)
      self.params = fromfile(self.file, 'i', self.numParams)

      self.nx,self.ny,self.nf = self.params[3:6]
      self.nxProcs,self.nyProcs = self.params[5:7]
   # end read_params

   def rewind(self):
      """Rewind the file to the start of the data (just past metadata)"""
      self.file.seek(4 * (self.numParams + 1))
   # end rewind

# end class PVReadSparse

