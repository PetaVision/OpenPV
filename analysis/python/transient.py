#!/usr/bin/python

# script to estimate the firing rate in L1 versus the background rate in retina
# copy this script to the location of your sandbox (STDP, marian, etc)

import os
import re     # regular expression module
import time

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

sys.path.append('/Users/manghel/Documents/workspace/PetaVision/analysis/python/')
import PVReadWeights as rw
import PVReadSparse as rs

path = '/Users/manghel/Documents/workspace/STDP/'

if len(sys.argv) < 6:
   print "usage: python transient.py timeSteps writeSteps dT minD start_flag"
   exit()

def modify_input(flag):
    print 'modify param.stdp for initFromLastFlag = %d' %  flag

    input = open(path + 'input/params.base','r')
    output = open(path + 'input/params.stdp','w')

    while 1:
        line = input.readline()
        if line == '':break

        if line.find('initFromLastFlag') >= 0:
            S = '   initFromLastFlag = ' + str(flag) + ';'
            output.write(S)
        else:
            output.write(line)

    input.close()
    output.close()
    
    return 0
# end modify_input

def compute_histogram(totalSteps, writeSteps, dT):
    infile = path + 'output/' + 'w0_last.pvp'
    output = open(path + 'output/w0_last_hist_' + str(totalSteps) + '.dat','w')

    time = totalSteps * dT
    histD = 100000 # make sure is larger than minD when this is first called
                   # abd we do not actually measure the hist distance between
                   # current weight distribution and previous weight distribution
    w = rw.PVReadWeights(infile)
    h = w.histogram()

    # write histogram
    for i in range(len(h)):
       output.write(str(h[i]) + '\n')
    output.close()

    # plot histogram
    fig = plt.figure(1)
    plt.subplot(2,2,1)
    plt.plot(np.arange(len(h)), h, 'o', color='y')
    plt.xlabel('WEIGHT BINS')
    plt.ylabel('COUNT')
    plt.title('Weight Histogram')
    plt.xlim(0, 256)
    plt.grid(True)
    plt.draw()

    # read and plot previous histogram
    # compute histogram distance

    if totalSteps-writeSteps > 0:
       input = open(path + 'output/w0_last_hist_' + str(totalSteps-writeSteps) + '.dat','r')
       h_old = np.zeros(256, dtype=int)
       dh = np.zeros(256,dtype=float)
       k=0;
       while 1:
          line = input.readline()
          if line == '':break
          h_old[k] = int(line);
          dh[k] = (1.0*abs((h[k]-h_old[k])))/max(h[k],h_old[k])
          k+=1   
       input.close()
       histD = max(dh)     # histogram distance
       print 'histD = %f\n' % (histD)
       plt.plot(np.arange(len(h_old)), h_old, 'o', color='r')
       plt.draw()

       #bx = fig.add_subplot(122, axisbg='darkslategray')
       plt.subplot(2,2,2)
       plt.plot(np.arange(len(dh)), dh, 'o', color='y')
       plt.xlabel('WEIGHT BINS')
       plt.ylabel('RELATIVE ERROR')
       plt.title('Weight Histogram RELATIVE ERROR')
       plt.xlim(0, 256)
       plt.grid(True)
       plt.hold(False)
       plt.draw()

       plt.subplot(2,2,3)
       plt.plot(time/1000, histD, 'o', color='b')
       plt.xlabel('TIME')
       plt.ylabel('HISTOGRAM ERROR')
       plt.title('Weight Histogram Error')
       plt.hold(True)
       plt.draw()

    return histD
    
# end compute_histogram

# time is the time when the rate estimation is done from the last 
# writeSteps spike records: we use the last rateSteps from timeSteps

def compute_rate(time, timeSteps, rateSteps, dT):

    infile = path + 'output/' + 'a1.pvp'
    output = open(path + 'output/rate.stdp','a')

    beginTime = (timeSteps - rateSteps)*dT
    endTime = timeSteps*dT

    s = rs.PVReadSparse(infile);
    rate = s.average_rate(beginTime,endTime) 

    output.write(str(time) + ' ' + str(rate) + '\n')
    output.close()

    # append rate subplot
    plt.figure(1)
    plt.subplot(2,2,4)
    plt.plot(time/1000, rate, 'o', color='b')
    plt.xlabel('TIME')
    plt.ylabel('AVERAGE RATE')
    plt.title('Firing Rate Evolution')
    plt.hold(True)
    plt.draw()

    return rate
# end compute_rate

""" 
Main code:
 
- modifies the params.stdp file to set initFromLastFlag

- run PV for writeSteps

- compute the histogram of the weights distribution.

- compare the histogram with the old histogram

"""

timeSteps = sys.argv[1]     # length of simulation (timeSteps)
writeSteps = sys.argv[2]    # estimate histogram every writeSteps
dT         = float(sys.argv[3])    # simulation time step
minD      = float(sys.argv[4])         # stoping condition
start_flag = int(sys.argv[5])    # starting flag

print '\ntimeSteps = %s writeSteps = %s dT = %f minD = %f start_flag = %d\n' % (timeSteps,writeSteps,dT, minD, start_flag)
totalSteps = 0;
histD = 1000;

while histD > minD and totalSteps < int(timeSteps):
    
    print '\ntotalSteps = %d \n' % totalSteps
    if totalSteps == 0:
       modify_input(start_flag)
    else:
       modify_input(1)

    #time.sleep(10)
    cmd = path + '/Debug/stdp -n ' + writeSteps + ' -p ' + path + '/input/params.stdp'
    #print cmd
    os.system(cmd)
    totalSteps = totalSteps + int(writeSteps)

    # compute, write, and plot histogram
    histD = compute_histogram(totalSteps, int(writeSteps), dT)

    # compute rate: arguments are: time, timeSteps, rateSteps, dT
    rate = compute_rate(totalSteps*dT,int(writeSteps), int(writeSteps),dT)
    print '\n average rate = %f \n' % rate

    # remove files
    if 0:
       cmd = 'rm ' + path + '/output/images/*'
       os.system(cmd)

       cmd = 'rm ' + path + '/output/*.pvp'
       os.system(cmd)

    
#time.sleep(10)
print 'pass control to plot'
plt.show()
