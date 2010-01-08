from numpy import *

class PVReadSparse(object):
   """A class to read PetaVision sparse activity files"""

   def __init__(self, filename):
      """Constructor: Open file and read in file metadata into params"""
      
      self.open(filename)
      self.read_params()
      self.time = 0.0
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
      print 'nx = %d ny = %d nf = %d' % (self.nx,self.ny,self.nf)
   # end read_params

   def rewind(self):
      """Rewind the file to the start of the data (just past metadata)"""
      self.file.seek(4 * (self.numParams ))
      
   # end rewind

   def average_rate(self,beginTime, endTime):
   
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

