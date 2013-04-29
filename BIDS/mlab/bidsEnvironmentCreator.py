import matplotlib.pyplot as plt
import numpy as np
import random

JITTER = False

world = np.zeros((256, 256));
nodex = range(2, 256, 4)
nodey = range(2, 256, 4)

#Jitter
if JITTER:
    jnode = [(y+random.randint(-3, 3), x+random.randint(-3, 3)) for y in nodey for x in nodex]
    jnode = [(y, x) for (y, x) in jnode if y>=0 and y<256 and x>=0 and x<256]
else:
    jnode = [(y, x) for y in nodey for x in nodex]


for coor in jnode:
    world[coor[0]][coor[1]] = 1

plt.gray()

plt.imshow(world)
plt.show()

