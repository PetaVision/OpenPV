"auc_average find the average of the auc and saves them"
import sys
import numpy as np


lenv = len(sys.argv)-1
print lenv

auc = np.zeros((lenv))
print auc


for i in range(lenv):
   aucp = np.array(np.genfromtxt(sys.argv[i+1], unpack=True))
   auc[i] = np.average(aucp)

print auc

   
np.savetxt('auc-line.txt', auc, fmt="%f")




