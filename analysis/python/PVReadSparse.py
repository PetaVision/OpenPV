import numpy as np
import PVConversions as conv

class PVReadSparse(object):
   """A class to read PetaVision sparse activity files"""

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
   
   def read_params(self):
      """Read the file metadata parameters"""
      self.file.seek(0)

      head = np.fromfile(self.file, 'i', 3)
      if head[2] != 2:
         print 'Incorrect file type ' + str(head[2])
         return

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

      self.time = np.fromfile(self.file, 'd', 1)

      #
      # figure out dt
      t0 = self.time
      rec = self.next_record()
      self.rewind()

      self.dt = self.time - t0
      self.time = t0
      self.timestep = 0

      #print 'nx = %d ny = %d nf = %d' % (self.nx,self.ny,self.nf)
   # end read_params

   def rewind(self):
      """Rewind the file to the start of the data (just past metadata)"""
      self.file.seek(4 * (self.numParams + 2))
   # end rewind

   def next_record(self):
      """Read the next record and return an ndarray"""
      self.time = np.fromfile(self.file, 'd', 1) [0]
      numItems = np.fromfile(self.file, 'i', 1) [0]
      if (numItems > 0):
         activity = np.fromfile(self.file, 'i', numItems)
      else:
         activity = []
      self.timestep += 1
      return activity
   # end next

   def minmax_k(self):
      """Returns the minimum and maximum k locations"""
      self.rewind()
      min = 999999999
      max = 0
      try:
         while True:
            r = self.next_record()
            for k in range(len(r)):
               if min > r[k]: min = r[k]
               if max < r[k]: max = r[k]
      except MemoryError:
         return (min,max)
      return (min,max)
   #end minmax_k

   def next_activity(self):
      """Return activity matrix for next time step"""
      A = np.zeros((self.nyg_ex, self.nxg_ex), int)
      rec = self.next_record()
      for k in range(len(rec)):
         kx = conv.kxPos(rec[k], self.nxg_ex, self.nyg_ex, self.nf)
         ky = conv.kyPos(rec[k], self.nxg_ex, self.nyg_ex, self.nf)
         A[ky,kx] = 1
      return A
   # end next_activity

   def avg_activity(self, begin, end):
      """Return total activity matrix for a given time period"""
      while self.time + self.dt < begin:
         print "skipping time=", self.time
         rec = self.next_record()
      t0 = self.time

      A = np.zeros((self.nyg_ex, self.nxg_ex))
      try:
         while self.time < end:
            A += self.next_activity()
      except MemoryError:
         dt = self.time - t0
         print "ts==%f te==%f dt==%f" %(t0,self.time,dt)

      dt = self.time - t0
 
      if dt > 0: return A/(.001*dt)
      else: return A
   # end avg_activity

   def average_rate(self, beginTime, endTime):
      self.rewind()
      average_rate = 0
      print 'beginTime = %f endTime = %f' % (beginTime,endTime)
      debug = 0

      # read transient time steps
      print 'read transient time steps'
      while self.time < beginTime:
         k = self.next_record()
         if len(k) > 0:
            if debug:
               print 'time = %f timestep = %i numItems = %i: ' % (self.time,self.timestep,len(k)),
               for i in range(len(k)):
                  print '%d ' % k[i],
               print '\n'
         else:
            if debug:
               print 'time = %f timestep = %i numItems = %i' % (self.time,self.timestep,len(k))
         
      # read time steps used to compute rate
      print 'read time steps to compute rate'
      while self.time < endTime:
         k = self.next_record()
         if len(k) > 0:
            average_rate += len(k)
            if debug:
               print 'time = %f timestep = %i numItems = %i: ' % (self.time,self.timestep,len(k)),
               for i in range(len(k)):
                  print '%d ' % k[i],
               print '\n'
         else:
            average_rate += 0
            if debug:
               print 'time = %f timestep = %i numItems = %i' % (self.time,self.timestep,len(k))
         
      average_rate = (average_rate * 1000.0) / ((endTime-beginTime)*self.nx*self.ny*self.nf)
      return average_rate

   # end average_rate

   def increment_rate(self,endTime):
      
      average_rate = 0
      beginTime = self.time
      print 'beginTime = %f endTime = %f' % (beginTime,endTime)
      debug = 0
         
      # read time steps used to compute rate
      print 'read time steps to compute rate'
      while self.time < endTime:
         k = self.next_record()
         if len(k) > 0:
            average_rate += len(k)
            if debug:
               print 'time = %f timestep = %i numItems = %i: ' % (self.time,self.timestep,len(k)),
               for i in range(len(k)):
                  print '%d ' % k[i],
               print '\n'
         else:
            average_rate += 0
            if debug:
               print 'time = %f timestep = %i numItems = %i' % (self.time,self.timestep,len(k))
         
      average_rate = (average_rate * 1000.0) / ((endTime-beginTime)*self.nx*self.ny*self.nf)
      return average_rate

   # end increment_rate

# end class PVReadSparse

