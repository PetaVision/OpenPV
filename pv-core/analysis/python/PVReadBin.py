import numpy as np
import PVConversions as conv

class PVReadBin(object):
   """A class to read PetaVision binary files"""

   def __init__(self, filename, extended):
      """Constructor: Open file and read in file metadata into params"""
      self.time = 0.0
      self.timestep = 0
      self.extended = extended
      self.open(filename)
      self.read_params()
   # end __init__

   def open(self, filename):
      """Open given filename"""
      self.filename = filename
      self.file = open(filename, mode='rb')
   # end open
   
   def next_record(self):
      """Read the next record and return an ndarray"""
      a = np.fromfile(self.file, 'f', self.numItems)
      self.timestep += 1
      return a
   # end next

   def read_params(self):
      """Read the file metadata parameters"""
      self.file.seek(0)

      head = np.fromfile(self.file, 'i', 3)
      print 'File type is ' + str(head[2])

      self.headerSize = head[0]
      self.numParams  = head[1] - 2  # two for time (a double)
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

      self.nxg_ex = self.nxGlobal
      self.nyg_ex = self.nyGlobal

      if self.extended:
         self.nxg_ex += 2*self.nPad
         self.nyg_ex += 2*self.nPad

      self.numItems = self.nxg_ex * self.nyg_ex

      self.timestep = 0
      self.time = np.fromfile(self.file, 'd', 1)

      #
      # figure out dt
      #rec = self.next_record()
      #t0 = self.time
      #rec = self.next_record()
      #t1 = self.time

      #self.dt = t1 - t0
      #self.time = t0

      #self.rewind()

      #print 'nx = %d ny = %d nf = %d' % (self.nx,self.ny,self.nf)
   # end read_params

   def rewind(self):
      """Rewind the file to the start of the data (just past metadata)"""
      self.file.seek(4 * (self.numParams + 2))
   # end rewind

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

