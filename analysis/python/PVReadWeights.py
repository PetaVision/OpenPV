import numpy as np
import math as math
import PVConversions as conv

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
      self.numWgtParams = 6
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
      w = np.zeros(len(bytes), dtype=float) + bytes/255.
      w = self.min + (self.max - self.min) * w   
      return w
   # end next

   def next_patch_bytes(self):
      """Read the next patch and return an ndarray"""
      nx,ny = np.fromfile(self.file, np.int16(), 2)
      count = nx*ny
      total = self.nxp * self.nyp * self.nfp

      bytes = np.fromfile(self.file, np.uint8(), total)

      self.patch += 1
      return bytes[0:count]
   # end next

   def normalize(self, w):
      a = math.sqrt( np.sum( w*w ) )
      if a != 0.0:
         w = w / a
      return w
   # end normalize

   def read_params(self):
      """Read the file metadata parameters"""
      self.file.seek(0)

      head = np.fromfile(self.file, 'i', 3)
      if head[2] != 3:  
         print 'Incorrect file type ' + str(head[2])
         return

      self.headerSize = head[0]
      self.numParams  = head[1] - 8  # 6 + two for time (a double)
      self.filetype   = head[2]

      self.file.seek(0)
      self.params = np.fromfile(self.file, 'i', self.numParams)

      self.nx,self.ny,self.nf = self.params[3:6]
      self.numRecords = self.params[6]
      self.recSize = self.params[7]
      self.dataSize = self.params[8]
      self.dataType = self.params[9]
      self.nxprocs,self.nyprocs = self.params[10:12]
      self.nxGlobal,self.nyGlobal = self.params[12:14]
      self.kx0,self.ky0 = self.params[14:16]
      self.nPad = self.params[16]
      self.nf = self.params[17]

      self.time = np.fromfile(self.file, 'd', 1)

      self.nxp,self.nyp,self.nfp = np.fromfile(self.file, 'i', 3)[0:3]
      self.min,self.max = np.fromfile(self.file, 'f', 2)[0:2]
      self.numPatches = np.fromfile(self.file, 'i', 1)[0]

      # define the numWeights
      self.patchSize = self.nxp * self.nyp * self.nfp
      self.numWeights = self.numPatches * self.patchSize

      #print 'finish reading file header'
   # end read_params

   def print_params(self):
      """Print the file metadata parameters"""

      print "numParams = %i" % (self.params[1]-8)  
      print "nx = %i ny = %i nf = %i" \
          % (self.params[3],self.params[4],self.params[5])
      print "numRecords = %i" % self.params[6]
      print "recSize = %i" %self.params[7]
      print "nPad = %i" % self.params[16]
      print "nf = %i " % self.params[17]
      print "time = %f" % self.time 
      print "nxp = %i nyp = %i nfp= %i" % (self.nxp,self.nyp,self.nfp)
      print "min = %i max = %i" % (self.min,self.max)
      print "numPatches = %i" % self.numPatches
   # end print_params

   def print_self_params(self):
      """Print the file metadata parameters"""

      print "numParams = %i" % self.numParams  
      print "nx = %i ny = %i nf = %i" % (self.nx,self.ny,self.nf)
      print "numRecords = %i" % self.numRecords
      print "recSize = %i" %self.recSize
      print "nPad = %i" % self.nPad 
      print "nf = %i " % self.nf 
      print "time = %d" % self.time 
      print "nxp = %i nyp = %i nfp= %i" % (self.nxp,self.nyp,self.nfp)
      print "min = %i max = %i" % (self.min,self.max)
      print "numPatches = %i" % self.numPatches
   # end print_params

   def rewind(self):
      """Rewind the file to the start of the data (just past metadata)"""
      self.file.seek(self.headerSize)
      self.patch = 0
   # end rewind

   def just_rewind(self):
      """Rewind the file to the start of the file"""
      self.file.seek(0)
      self.patch = 0
   # end just_rewind

   def clique_locations(self, count, wVal):
      """Return the locations of cliques of a given size"""
      self.rewind()
      x = []
      y = []
      h = np.zeros(1 + self.patchSize, dtype=int)
      for k in range(self.numPatches):
         b = self.next_patch_bytes()
         if len(b) == self.patchSize:
            csize = self.clique_size(b, wVal)
            if csize == count:
               nxg = self.nxGlobal
               nyg = self.nyGlobal
               nxb = self.nxprocs
               nyb = self.nyprocs
               kx = conv.kxBlockedPos(k, nxg, nyg, self.nf, nxb, nyb)
               ky = conv.kyBlockedPos(k, nxg, nyg, self.nf, nxb, nyb)
               x.append(kx)
               y.append(ky)
      return np.array([x,y])
   # end clique_locations

   def clique_size(self, b, wVal):
      """Return the number of weights in a patch greater than wVal"""
      count = 0
      for k in range(len(b)):
         if b[k] > wVal:
            count += 1
      return count
   # end clique_size

   def clique_histogram(self, wVal):
      """Return a histogram of clique size"""
      self.rewind()
      h = np.zeros(1 + self.patchSize, dtype=int)
      for p in range(self.numPatches):
         b = self.next_patch_bytes()
         if len(b) == self.patchSize:
            csize = self.clique_size(b, wVal)
            h[csize] += 1
      return h
   # end clique_histogram

   def histogram(self):
      self.rewind()
      h = np.zeros(256, dtype=int)
      for p in range(self.numPatches):
         b = self.next_patch_bytes()
         for k in range(len(b)):
            h[b[k]] += 1
      return h
   # end histogram

   def next_record(self):
      self.read_params()
      self.print_params()

      r = np.zeros(self.numWeights,dtype = float32) 

      for p in range(self.numPatches):
         w = self.next_patch()
         for k in range(self.patchSize):
           r[p*self.patchSize + k ] = w[k]
         if p == -1:
            for k in range(p*self.patchSize,(p+1)*self.patchSize):
               print r[k],
            print
         #s = raw_input('--> ')
         self.patch += 1

      return r
   # end next_record

   def valuesAt(self, k):
      """Return an ndarray of values at index k in each time slice (record)"""
      n = 0
      v = []
      self.rewind()
      try:
         while True:
            v.append( self.next_record() [k] ) # should be next_patch?
            n += 1
      except:
         print "Finished reading, read", n, "records"

      a = np.zeros(len(v))
      for i in range(len(v)):
         a[i] = v[i]

      return a
   # end valuesAt

# end class PVReadWeights

