#!/usr/bin/python

# Script to compute the temporal correlations of the weights between 
# the Retina and the L1 layer;
# NOTE: copy this script to the location of your sandbox (STDP, marian, etc)

import os
import re     # regular expression module
import time
import string

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

sys.path.append('/Users/manghel/Documents/workspace/PetaVision/analysis/python/')
import PVReadWeights as rw
import PVReadSparse as rs

path = '/Users/manghel/Documents/workspace/STDP/'

# find strParam and set its value to valParam
def modify_input(strParam, valParam):
    print 'modify param.stdp for ' + strParam + ' = ' + valParam

    input = open(path + 'input/params.base','r')
    output = open(path + 'input/params.stdp','w')

    while 1:
        line = input.readline()
        if line == '':break

        if line.find(strParam) >= 0:
            S = strParam + ' = ' + valParam + ';'
            output.write(S)
        else:
            output.write(line)

    input.close()
    output.close()
    
    return 0
# end modify_input

def compute_correlations(timeSteps, writeStep, dT, p):

    infile = path + 'output/' + 'w0_post.pvp'

    w = rw.PVReadWeights(infile) # opens file and reads params!!
    print 'numWeights = ' + str(w.numWeights) + ' patchSize = ' + str(w.patchSize)
    w.print_params()
    w.just_rewind()
    T = int(timeSteps*dT)/writeStep
    weights = np.zeros((T,w.patchSize),dtype=np.float32)
    # read the first record (time 0.5)
    r = w.next_record() # read header and next record (returns numWeights array)

    # from now on, with writeSteps = 1, we write weights every integer time
    n = 0
    try:
         while True:
            r = w.next_record() # read header and next record (returns numWeights array)
            m = 0
            print str(n+1) + ': ',
            for k in range(p*w.patchSize,(p+1)*w.patchSize):
               weights[n,m] = r[k]
               m += 1
               print r[k],
            print
            n+=1
            #s = raw_input('--> ')
    except:
         print "Finished reading, read", n, "records"

    # plot weights evolution
    sym = np.array(['r','b','g','r','b','g','r','b','g','r','b','g','r','b','g','r'])
    fig = plt.figure(1)
    plt.subplot(2,1,1)
    for k in range(w.patchSize):
       plt.plot(np.arange(T), weights[:,k], '-o', color=sym[k])

    plt.xlabel('Time')
    plt.ylabel('Weights')
    plt.title('Weights Evolution')
    plt.hold(True)
    plt.draw()

    # compute and plot correlations
    plt.subplot(2,1,2)
    for k in range(w.patchSize):
       plt.acorr(weights[:,k], normed=True, maxlags=3,linestyle = 'solid', color = sym[k])
    plt.xlabel('Time')
    plt.ylabel('Corr')
    plt.title('Weights Autocorrelations')
    plt.hold(True)
    plt.draw()

# end compute_correlations

""" 
Main code:
 
- modifies the params.stdp file to set writeSteps

- run PV for timeSteps

- compute the weight correlations.


"""

if len(sys.argv) < 6:
   print "usage: python weights_corr.py timeSteps writeSteps numLayers dT run_flag"
   print "where: - timeSteps is the number of steps used to compute"
   print "         the weights correlations"
   print "       - writeSteps is the number of steps used to write the post-synaptic"
   print "         weights"
   print "       - numLayers is the number of neural layers"
   print "       - dT is the duration of each time step (in milliseconds)"
   print "       - run_flag if 1 we run the simulation for timeSteps and then"
   print "         compute correlations; if 0 we only compute correlations"
   exit()


timeSteps  = int(sys.argv[1])    
writeStep = sys.argv[2]         
numLayers  = int(sys.argv[3])    
dT         = float(sys.argv[4])   
run_flag   = int(sys.argv[5])

print '\ntimeSteps = %d writeStep = %s numLayers = %d dT = %f \n' \
     % (timeSteps,writeStep,numLayers,dT)

if run_flag == 1:
   modify_input('writeStep',writeStep)   
   s = raw_input('params.stdp was modified ')
   cmd = path + '/Debug/stdp -n ' + str(timeSteps) + ' -p ' + path + '/input/params.stdp'
   print cmd
   os.system(cmd)
    

# compute, write, and plot correlations
p = 635 # patch to analyze
compute_correlations(timeSteps, int(writeStep), dT, p)


print 'pass control to plot'
plt.show()
