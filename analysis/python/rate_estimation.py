#!/usr/bin/python

# script to estimate the firing rate in L1 versus the background rate in retina
# copy this script to the location of your sandbox (STDP, marian, etc)

import os
import re     # regular expression module
import time

import sys
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab

sys.path.append('/Users/manghel/Documents/workspace/PetaVision/analysis/python/')
import PVReadWeights as rw
import PVReadSparse as rs

path = '/Users/manghel/Documents/workspace/STDP/'

if len(sys.argv) < 4:
   print "usage: python rate_estimation timeSteps rateSteps dT"
   exit()

def modify_input(p):
    #print 'modify param.stdp for noiseOffFreq = %f' %  p

    input = open(path + 'input/params.base','r')
    output = open(path + 'input/params.stdp','w')

    while 1:
        line = input.readline()
        if line == '':break

        if line.find('noiseOffFreq') >= 0:
            S = '   noiseOffFreq = ' + str(p) + ';\n'
            output.write(S)
        else:
            output.write(line)

    input.close()
    output.close()
    
    return 0
# end modify_input

def compute_rate(p,timeSteps, rateSteps, dT):

    infile = path + 'output/' + 'a1.pvp'
    output = open(path + 'output/rate.stdp','a')

    beginTime = (timeSteps-rateSteps)*dT
    endTime = timeSteps*dT

    s = rs.PVReadSparse(infile);
    rate = s.average_rate(beginTime,endTime)

    output.write(str(p) + ' ' + str(rate) + '\n')
    output.close()

    return rate
# end compute_rate

def compute_histogram(p):
    infile = path + 'output/' + 'w0_last.pvp'
    output = open(path + 'output/w0_last_hist_' + str(p) + '.dat','w')

    w = rw.PVReadWeights(infile)
    h = w.histogram()

    for i in range(len(h)):
       output.write(str(h[i]) + '\n')

    output.close()

# end compute_histogram


""" 
Main code:
 
- modifies the params.stdp file to set the retina
background noise.

- compute time-averaged firing rate in L1.

- compute the histogram of the weights distribution.

"""

p = 10                      # starting background retina noise (Hz)
timeSteps = sys.argv[1]     # length of simulation (timeSteps)
rateSteps = sys.argv[2]     # asymptotic time steps used to compute rate
dT        = sys.argv[3]     # simulation time step interval
print '\ntimeSteps = %s rateSteps = %s dT = %s \n' % (timeSteps,rateSteps,dT)

while p <= 100:
    print 'run model for noiseOffFreq = %f' % p
    modify_input(p)
    #time.sleep(10)
    cmd = path + '/Debug/stdp -n ' + timeSteps + ' -p ' + path + '/input/params.stdp'
    #print cmd
    os.system(cmd)
    rate = compute_rate(p, float(timeSteps), float(rateSteps), float(dT) )
    print ' p = %f rate = %f \n' % (p,rate) 
    
    # compute histogram
    compute_histogram(p)

    # remove files
    cmd = 'rm ' + path + '/output/images/*'
    os.system(cmd)

    cmd = 'rm ' + path + '/output/*.pvp'
    os.system(cmd)

    p +=  100


