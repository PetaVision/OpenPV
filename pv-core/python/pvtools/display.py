from .readpvpfile import readpvpfile
from .pv_object import PV_Object
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def interpret(arg):
   if type(arg) is str:
      arg = readpvpfile(arg)
   assert type(arg) is PV_Object
   return arg

def view(data,frame=0):
   assert type(frame) is int
   data = interpret(data)

   if data.header['filetype'] == 4:
      plt.imshow(data[frame].values,
                 interpolation='nearest')
      plt.axis('off')
      plt.show()

   if data.header['filetype'] == 5:
         axes = np.ceil(np.sqrt(data.header['nf']))
         for patch in range(data.header['nf']):
            plt.subplot(axes,axes,patch+1)
            plt.imshow(data[frame].values[0][:,:,:,patch],
                       interpolation='nearest')
            plt.axis('off')
         plt.show()

def showErrorPlot(image, *args):
   image = interpret(image)
   if args:
      for arg in args:
         plt.figure()
         plt.plot(image.getError(interpret(arg)))
         plt.show()
   else:
      plt.figure()
      plt.plot(image.getError())
      plt.show()

def showNumActivePlot(data):
   data = inpterpret(data)
   plt.figure()
   plt.plot(image.getActive())
   plt.show()
   
def showSparsityPlot(data):
   data = interpret(data)
   plt.figure()
   plt.plot(image.getPercentActive())
   plt.show()
