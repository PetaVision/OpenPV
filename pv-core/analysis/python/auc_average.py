"auc_average find the average of the auc and saves them"
import sys
import numpy as np


auc1 = sys.argv[1]
auc2 = sys.argv[2]
auc3 = sys.argv[3]
auc4 = sys.argv[4]






aucd1 = np.array(np.genfromtxt(auc1, unpack=True))
aucd2 = np.array(np.genfromtxt(auc2, unpack=True))
aucd3 = np.array(np.genfromtxt(auc3, unpack=True))
aucd4 = np.array(np.genfromtxt(auc4, unpack=True))


aucl1 = int(np.shape(aucd1)[0])
aucl2 = int(np.shape(aucd2)[0])
aucl3 = int(np.shape(aucd3)[0])
aucl4 = int(np.shape(aucd4)[0])

if aucl1 > 5:
   dif = aucl1 - 10
   for j in range(dif):
      aucd1 = np.delete(aucd1, -1, axis=0)

aucd1 = np.average(aucd1)
aucd2 = np.average(aucd2)
aucd3 = np.average(aucd3)
aucd4 = np.average(aucd4)



result = np.vstack((aucd4, aucd3))
result = np.vstack((result, aucd2))
result = np.vstack((result, aucd1))

np.savetxt('auc-line.txt', result, fmt="%f")

print result




