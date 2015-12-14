from readpvpfile import readpvpfile
from pv_object import PV_Object
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def view(data,frame=0):
   assert type(frame) is int
   if type(data) is str:
      data = readpvpfile(data)
   assert type(data) is PV_Object

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
