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
      self.numWgtParams = 6
      print 'open ' + filename
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

      self.patch += 1   

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
      #print 'read file header'

      head = fromfile(self.file, 'i', 3)
      #print str(head[0]) + ' ' + str(head[1]) + ' ' + str(head[2])
      if head[2] != 3:  
         print 'Incorrect file type ' + str(head[2])
         return

      # self.numWgtParams = 6

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

      self.wgtParams = fromfile(self.file, 'i', self.numWgtParams)

      self.nxp,self.nyp,self.nfp = self.wgtParams[0:3]
      self.min,self.max = self.wgtParams[3:5]
      self.numPatches = self.wgtParams[5]

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
      print "numRecords = %i" % self.params[8]
      print "recSize = %i" %self.params[9]
      print "nPad = %i" % self.params[16]
      print "nf = %i " % self.params[17]
      print "time = %f" % self.time 
      print "nxp = %i nyp = %i nfp= %i" \
         % (self.wgtParams[0],self.wgtParams[1],self.wgtParams[2])
      print "min = %i max = %i" % (self.wgtParams[3],self.wgtParams[4])
      print "numPatches = %i" % self.wgtParams[5]
   # end print_params

   def print_params_old(self):
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


   def read_header(self):
      """Read the header of each record"""

      self.params = fromfile(self.file, 'i', self.numParams)

      self.time = fromfile(self.file, 'd', 1)

      self.wgtParams = fromfile(self.file, 'i', self.numWgtParams)

      self.patch = 0

   # end read_header

 
   def rewind(self):
      """Rewind the file to the start of the data (just past metadata)"""
      self.file.seek(self.headerSize)
   # end rewind

   def just_rewind(self):
      """Rewind the file to the start of the file"""
      self.file.seek(0)

   # end just_rewind

   def histogram(self):
      self.rewind()
      h = zeros(256, dtype=int)
      for p in range(self.numPatches):
         b = self.next_patch_bytes()
         for k in range(len(b)):
            h[b[k]] += 1
      return h
   # end histogram

   def next_record(self):
      self.read_header()
      #self.print_params()
      #r = []
      r = zeros(self.numWeights,dtype = float32) 
      #print r.shape
      for p in range(self.numPatches):
         #print p
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

   # NOTE: self.next_record was not defined.
   # my definition might be different from the one Craig had in mind
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

      a = zeros(len(v))
      for i in range(len(v)):
         a[i] = v[i]

      return a
   # end valuesAt

# end class PVReadWeights

