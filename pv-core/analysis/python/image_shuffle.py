import sys
import numpy as np
import matplotlib.image as mpimg
import numpy.random as npr
import Image

mode = 'L'

for i in range(len(sys.argv)-1):
   i+=1

   mi = mpimg.imread(sys.argv[i])


   x = np.shape(mi)[0]
   mi = np.reshape(mi, (x*x))
   mi = npr.permutation(mi)
   mi = np.reshape(mi, (x,x))

   size = np.shape(mi)
 
   imNew=Image.new(mode , size)
   data = np.ravel(mi)
   data = np.floor(data * 256)

   imNew.putdata(data)
   imNew.save("../new-images/%s" %(sys.argv[i]))

