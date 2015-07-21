#!/usr/bin/python

# Script to estimate:
#     - the evolution of the STDP controlled weights between the Retina and 
#       the L1 layer;
#     - the rate of change of the histogram distance between two weight 
#       distributions separated by writeSteps simulation steps;
#     - the evolution of the number of weights in the smalest and largest
#       histogram bin;
#     - the evolution of the average (space and time) firing rate in each layer;
# NOTE: copy this script to the location of your sandbox (STDP, marian, etc)

import os
import re     # regular expression module
import time
import string

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab



MAC = True
if MAC == True:
   path = '/Users/manghel/Documents/workspace/STDP/'
   sys.path.append('/Users/manghel/Documents/workspace/PetaVision/analysis/python/')
else:
   path = '/nh/home/manghel/petavision/workspace.pv/STDP/'
   sys.path.append('/nh/home/manghel/petavision/workspace.pv/PetaVision/analysis/python/')
import PVReadWeights as rw
import PVReadSparse as rs

def write_histogram(h,conn, time):	  
    
    output = open(path + 'data3/w' + str(conn) + '_hist_' + str(time) + '.dat','w')
    for i in range(len(h)):
      output.write(str(h[i]) + '\n')
    output.close()

# end write_histogram

def plot_histogram(h, conn, time):

   fig = plt.figure(1)
   plt.subplot(2,2,1)
   plt.plot(np.arange(len(h)), h, '-', color='b')
   plt.xlabel('Weight Bins')
   plt.ylabel('Count')
   plt.title('Weight Histogram')
   plt.xlim(0, 256)
   plt.grid(True)
   plt.hold(True)
   plt.draw()


# end plot histogram
    
def plot_small_large_bins(hmin, hmax, conn, time):
    
  # evolution of the smallest and largest weight bins
  plt.subplot(2,2,2)
  plt.plot(time/1000, hmin, 'o', color='b')
  plt.xlabel('Time')
  plt.ylabel('Smallest/Largest weight bins')
  plt.title('Evolution of small/large weights')
  plt.hold(True)
  plt.plot(time/1000, hmax, 'o', color='r')
  plt.draw()
  output = open(path + 'output/small_large_bins.dat','a')
  output.write(str(time) + ' ' + str(h[0]) + ' ' + str(h[255]) + '\n')
  output.close()
    
# end plot small and large bins    

def compute_histograms(conn, totalSteps, writeSteps, dT):

    infile = path + 'output/' + 'w' + str(conn) + '_post.pvp'

    print 'compute histogram evolution for connection ' + str(conn)

    w = rw.PVReadWeights(infile)

    n = 0
    try: 
       while True:	
          h = w.histogram()
          n += 1

	  # write histogram
    	  write_histogram(h,conn,w.time)

	  # plot histogram
          plot_histogram(h,conn,w.time)

          # plot small and large bins
          plot_small_large_bins(h[0], h[255], conn, w.time)

    except:
       print 'Finished reading ' + str(n) + 'records: time = ' + str(w.time)
       s = raw_input('--> ')

# end compute_histograms

# 

def compute_rate(layer, totalSteps, writeStep, dT):

    sym = np.array(['r','b','g'])
    rate = np.zeros(numLayers,dtype=float)
    

    infile = path + 'output/' + 'a' + str(layer) + '.pvp'
    output = open(path + 'data3/rate' + str(layer) + '.stdp','a')

    s = rs.PVReadSparse(infile);

    endTime = writeStep*dT

    try:
       while True:
           rate = s.increment_rate(endTime) 
           output.write(str(endTime) + ' ' + str(rate) + '\n')
           # append rate subplot
           plt.figure(1)
           plt.subplot(2,2,4)
           plt.plot(endTime/1000, rate, 'o', color=sym[layer])
           plt.xlabel('Time')
           plt.ylabel('Average Rate')
           plt.title('Firing Rate Evolution')
           plt.hold(True)
           plt.draw()
           endTime += writeStep*dT
    except:
       print 'finished reading rate: time = ' + str(s.time)
       output.close()

# end compute_rate

""" 
Main code:
 
- modifies the params.stdp file to set initFromLastFlag

- run PV for runSteps

- compute the histogram of the weights distribution.

- compare the histogram with the old histogram

"""

if len(sys.argv) < 6:
   print "usage: python read_transient.py totalSteps writeStep numLayers numConns dT"
   print "where: - totalSteps is the number of simulation steps "
   print "where: - writeStep is the writing time "
   print "       - numLayers is the number of neural layers"
   print "       - numConns is the number of (STDP) connections"
   print "       - dT is the duration of each time step (in milliseconds)"
   exit()


totalSteps = int(sys.argv[1])
writeStep  = int(sys.argv[2])                     
numLayers  = int(sys.argv[3]) 
numConns   = int(sys.argv[4])   
dT         = float(sys.argv[5])  



print '\ntotalSteps = %d writeStep = %d numLayers = %d numConns = %d dT = %f  \n' \
     % (totalSteps,writeStep,numLayers,numConns,dT)

s = raw_input('--> ')

for conn in range(numConns):
   compute_histograms(conn, totalSteps,writeStep,dT)

for layer in range(numLayers):
   print 'rate evolution for layer ' + str(layer)
   #compute_rate(layer,totalSteps,writeStep,dT)



print 'pass control to plot'
plt.show()
