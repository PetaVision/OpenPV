"""
Print values of a given weight patch
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import PVReadWeights as rw

if len(sys.argv) < 3:
   print "usage: print_patch filename k_index"
   sys.exit()

k_loc = int(sys.argv[2])
w = rw.PVReadWeights(sys.argv[1])

for k in range(k_loc + 1):
   p_bytes = w.next_patch_bytes()

p = np.zeros(w.patchSize)

for k in range(w.patchSize):
   p[k] = p_bytes[k]

p = p.reshape((w.nyp, w.nxp))
print p
print w.min, w.max

