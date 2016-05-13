import numpy as np
import pdb
import matplotlib.pyplot as plt


momentumFilename = "/home/sheng/workspace/OpenPV/PVSystemTests/MomentumLCATest/output/total_energy.txt"
regFilename = "/home/sheng/workspace/OpenPV/PVSystemTests/LCATest/output/total_energy.txt"


f = open(momentumFilename, 'r')
momentumLines = f.readlines()
f.close()

f = open(regFilename, 'r')
regLines = f.readlines()
f.close()

momentumEnergy = [float(m.split(',')[3]) for m in momentumLines]
regEnergy = [float(m.split(',')[3]) for m in regLines]

plt.figure()
plt.plot(momentumEnergy, 'r')
plt.plot(regEnergy, 'b')
plt.show()






