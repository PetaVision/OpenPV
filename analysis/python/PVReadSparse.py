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
      #print 'nx = %d ny = %d nf = %d' % (self.nx,self.ny,self.nf)
   # end read_params

   def rewind(self):
      """Rewind the file to the start of the data (just past metadata)"""
      self.file.seek(4 * (self.numParams + 1))
   # end rewind

   def average_rate(self,timeSteps, rateSteps,dT):
   
      self.timeSteps = timeSteps
      self.rateSteps = rateSteps
      self.dT = dT    # 0.5; integration time step (miliseconds)
      self.rewind()
      average_rate = 0
      #print 'timeSteps = %d rateSteps = %d' % (self.timeSteps,self.rateSteps)

      # read transient time steps
      #print 'read transient time steps'
      while self.timestep < (self.timeSteps-self.rateSteps):
         #k = self.next_record()
         numItems = fromfile(self.file, 'i', 1)[0]
         if numItems > 0:
            #print 'timestep = %d numItems = %d: ' % (self.timestep,numItems),
            k = fromfile(self.file, 'i', numItems)
            #for i in range(len(k)):
            #   print '%d ' % k[i],
            #print '\n'
         else:
            average_rate += 0
            #print 'timestep = %d numItems = %d' % (self.timestep,numItems)
         self.timestep += 1
         
      self.timestep = 0
      # read time steps used to compute rate
      #print 'read time steps to compute rate'
      while self.timestep < self.rateSteps:
         #k = self.next_record()
         numItems = fromfile(self.file, 'i', 1)[0]
         if numItems > 0:
            #print 'timestep = %d numItems = %d: ' % (self.timestep,numItems),
            k = fromfile(self.file, 'i', numItems)
            #for i in range(len(k)):
            #   print '%d ' % k[i],
            #print '\n'
            average_rate += len(k)
         else:
            average_rate += 0
            #print 'timestep = %d numItems = %d' % (self.timestep,numItems)
         self.timestep += 1
         
      average_rate = (average_rate * 1000.0) / (self.timestep*self.nx*self.ny*self.nf*self.dT)
      return average_rate

   # end average_rate

# end class PVReadSparse

