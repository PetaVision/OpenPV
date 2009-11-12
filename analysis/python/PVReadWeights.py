from numpy import *

class PVReadWeights(object):
   """A class to read PetaVision binary weight files"""

   def __init__(self, filename):
      """Constructor: Open file and read in file metadata into params"""
      
      '''initialize'''
      self.numParams = 0
      self.params = ()
      self.nx = 0
      self.ny = 0
      self.nf = 0
      self.numItems = 0

      self.open(filename)
      self.read_params()

      '''current patch'''
      self.patch = 0

   # end __init__

   def open(self, filename):
      """Open given filename"""
      self.filename = filename
      self.file = open(filename, mode='rb')
   # end open
   
   def next_patch(self):
      """Read the next patch and return an ndarray"""

      bytes = self.next_patch_bytes()

      w = zeros(len(bytes), dtype=float) + bytes/255.
      w = self.min + (self.max - self.min) * w      

      return w
   # end next

   def next_patch_bytes(self):
      """Read the next patch and return an ndarray"""
      
      nx,ny = fromfile(self.file, int16(), 2)
      count = nx*ny
      total = self.nxp * self.nyp * self.nfp

      bytes = fromfile(self.file, uint8(), total)

      self.patch += 1
      return bytes[0:count]
   # end next

   def read_params(self):
      """Read the file metadata parameters"""

      head = fromfile(self.file, 'i', 3)
      if head[2] != 3:
         print "Incorrect file type"
         return

      numWgtParams = 6

      self.headerSize = head[0]
      self.numParams  = head[1] - 8  # 6 + two for time (a double)

      self.file.seek(0)
      self.params = fromfile(self.file, 'i', self.numParams)

      self.nx,self.ny,self.nf = self.params[3:6]
      self.nxprocs,self.nyprocs = self.params[6:8]
      self.numRecords = self.params[8]
      self.recSize = self.params[9]
      self.elemSize = self.params[10]
      self.dataType = self.params[11]
      self.nxBlocks,self.nyBlocks = self.params[12:14]
      self.nxGlobal,self.nyGlobal = self.params[14:16]
      self.kx0,self.ky0 = self.params[14:16]
      self.nPad = self.params[16]
      self.nf = self.params[17]

      self.time = fromfile(self.file, 'd', 1)

      self.wgtParams = fromfile(self.file, 'i', numWgtParams)

      self.nxp,self.nyp,self.nfp = self.wgtParams[0:3]
      self.min,self.max = self.wgtParams[3:5]
      self.numPatches = self.wgtParams[5]
   # end read_params

   def rewind(self):
      """Rewind the file to the start of the data (just past metadata)"""
      self.file.seek(self.headerSize)
   # end rewind

   def histogram(self):
      self.rewind()
      h = zeros(256, dtype=int)
      for p in range(self.numPatches):
         b = self.next_patch_bytes()
         for k in range(len(b)):
            h[b[k]] += 1
      return h
   # end histogram

   def valuesAt(self, k):
      """Return an ndarray of values at index k in each time slice (record)"""
      n = 0
      v = []
      self.rewind()
      try:
         while True:
            v.append( self.next_record() [k] )
            n += 1
      except:
         print "Finished reading, read", n, "records"

      a = zeros(len(v))
      for i in range(len(v)):
         a[i] = v[i]

      return a
   # end valuesAt

# end class PVReadBin

