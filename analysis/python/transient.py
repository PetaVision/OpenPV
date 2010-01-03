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

sys.path.append('/Users/manghel/Documents/workspace/PetaVision/analysis/python/')
import PVReadWeights as rw
import PVReadSparse as rs

path = '/Users/manghel/Documents/workspace/STDP/'

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

# read the last histogram from the end of the previous simulation
# plot it
def read_histogram(totalSteps):

    # read last histogram
    input = open(path + 'output/w0_last_hist_' + str(totalSteps) + '.dat','r')
    h = np.zeros(256, dtype=int)
    k=0;
    while 1:
       line = input.readline()
       if line == '':break
       h[k] = int(line);
       k+=1   
    input.close()

    # plot histogram
    fig = plt.figure(1)
    plt.subplot(2,2,1)
    plt.plot(np.arange(len(h)), h, 'o', color='r')
    plt.draw()

# read and plot evolution of average firing rate
def read_rate(numLayers):

    sym = np.array(['r','b','g'])
    # initialize rate subplot
    plt.figure(1)
    plt.subplot(2,2,4)

    for i in range(numLayers):
       # read rate evolution 
       input = open(path + 'output/rate' + str(i) + '.stdp','r')

       while 1:
          line = input.readline()
          if line == '':break
          data = string.split(line,' ')
          time = eval(data[0])
          rate = eval(data[1])
          plt.plot(time/1000,rate,'o',color=sym[i])
       
       input.close()

    plt.xlabel('Time')
    plt.ylabel('Average Rate')
    plt.title('Firing Rate Evolution')
    plt.hold(True)
    plt.draw()

# end read_rate


# read and plot evolution of histogram distance
def read_histD():

   # initialize histogram distance subplot
    plt.figure(1)
    plt.subplot(2,2,3)

    # read rate evolution 
    input = open(path + 'output/histD.stdp','r')

    while 1:
       line = input.readline()
       if line == '':break
       data = string.split(line,' ')
       time = eval(data[0])
       histD = eval(data[1])
       plt.plot(time/1000,histD,'o',color='b')
       
    input.close()

    plt.xlabel('Time')
    plt.ylabel('Histogram Distance')
    plt.title('Weight Histogram Distance')
    plt.hold(True)
    plt.draw()
    return histD


# read and plot evolution of histogram distance
def read_small_large_bins():

   # initialize small/large bins evolution subplot
    plt.figure(1)
    plt.subplot(2,2,2)

    # read rate evolution 
    input = open(path + 'output/small_large_bins.dat','r')

    while 1:
       line = input.readline()
       if line == '':break
       data = string.split(line,' ')
       time = eval(data[0])
       small_bin = eval(data[1])
       large_bin = eval(data[2])
       plt.plot(time/1000,small_bin,'o',color='b')
       plt.plot(time/1000,large_bin,'o',color='r')

    input.close()

    plt.xlabel('Time')
    plt.ylabel('Smallest/Largest weight bins')
    plt.title('Evolution of small/large weights')
    plt.hold(True)
    plt.draw()

def compute_histogram(totalSteps, writeSteps, dT):

    binDistance = 0  # when 1 plots the relative bin distance

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
    plt.hold(False)
    plt.plot(np.arange(len(h)), h, 'o', color='y')
    plt.xlabel('Weight Bins')
    plt.ylabel('Count')
    plt.title('Weight Histogram')
    plt.xlim(0, 256)
    plt.grid(True)
    plt.hold(True)
    plt.draw()

    # evolution of the smallest and largest weight bins
    plt.subplot(2,2,2)
    plt.plot(time/1000, h[0], 'o', color='b')
    plt.xlabel('Time')
    plt.ylabel('Smallest/Largest weight bins')
    plt.title('Evolution of small/large weights')
    plt.hold(True)
    plt.plot(time/1000, h[255], 'o', color='r')
    plt.draw()
    output = open(path + 'output/small_large_bins.dat','a')
    output.write(str(time) + ' ' + str(h[0]) + ' ' + str(h[255]) + '\n')
    output.close()

    # read and plot previous histogram
    # compute histogram distance

    if totalSteps-writeSteps > 0:
       output = open(path + 'output/histD.stdp','a')
       input = open(path + 'output/w0_last_hist_' + str(totalSteps-writeSteps) + '.dat','r')
       h_old = np.zeros(256, dtype=int)
       dh = np.zeros(256,dtype=float)
       k=0;
       while 1:
          line = input.readline()
          if line == '':break
          h_old[k] = int(line);
          maxH = max(h[k],h_old[k])
          if maxH > 0:
             dh[k] = (1.0*abs((h[k]-h_old[k])))/maxH
          else:
             dh[k] = -1
          k+=1   
       input.close()
       histD = (max(dh)*1000.0)/writeSteps     # histogram distance per unit time (s)
       print 'histD = %f\n' % (histD)
       output.write(str(time) + ' ' + str(histD) + '\n')
       output.close()

       # overplot old histogram
       plt.subplot(2,2,1)
       plt.plot(np.arange(len(h_old)), h_old, 'o', color='r')
       plt.xlim(0, 256)
       #plt.hold(False)
       plt.draw()

       #bx = fig.add_subplot(122, axisbg='darkslategray')
       if binDistance:
          plt.subplot(2,2,2)
          plt.plot(np.arange(len(dh)), dh, 'o', color='y')
          plt.xlabel('Weight Bins')
          plt.ylabel('Relative Bin Distance')
          plt.title('Weight Histogram Bin Distance')
          plt.xlim(0, 256)
          plt.grid(True)
          plt.hold(False)
          plt.draw()


       # evolution of histogram distance per unit time
       plt.subplot(2,2,3)
       plt.plot(time/1000, histD, 'o', color='b')
       plt.xlabel('Time')
       plt.ylabel('Histogram Distance')
       plt.title('Weight Histogram Distance')
       plt.hold(True)
       plt.draw()

    return histD
    
# end compute_histogram

# time is the time when the rate estimation is done from the last 
# writeSteps spike records: we use the last rateSteps from timeSteps

def compute_rate(numLayers, time, timeSteps, rateSteps, dT):

    sym = np.array(['r','b','g'])
    rate = np.zeros(numLayers,dtype=float)
    for i in range(numLayers):

       infile = path + 'output/' + 'a' + str(i) + '.pvp'
       output = open(path + 'output/rate' + str(i) + '.stdp','a')

       beginTime = (timeSteps - rateSteps)*dT
       endTime = timeSteps*dT

       s = rs.PVReadSparse(infile);
       rate[i] = s.average_rate(beginTime,endTime) 

       output.write(str(time) + ' ' + str(rate[i]) + '\n')
       output.close()

       # append rate subplot
       plt.figure(1)
       plt.subplot(2,2,4)
       plt.plot(time/1000, rate[i], 'o', color=sym[i])
       plt.xlabel('Time')
       plt.ylabel('Average Rate')
       plt.title('Firing Rate Evolution')
       plt.hold(True)
       plt.draw()

    return rate
# end compute_rate

""" 
Main code:
 
- modifies the params.stdp file to set initFromLastFlag

- run PV for runSteps

- compute the histogram of the weights distribution.

- compare the histogram with the old histogram

"""

if len(sys.argv) < 7:
   print "usage: python transient.py totalSteps timeSteps writeSteps numLayers dT minD"
   print "where: - totalSteps is the number of time steps up to this run"
   print "       - timeSteps (> totalSteps) is the number of steps at the"
   print "         end of this simulation"
   print "       - runSteps is the number of steps used to segment the"
   print "         current simulation (also used to define the firing rate)"
   print "       - numLayers is the number of neural layers"
   print "       - dT is the duration of each time step (in milliseconds)"
   print "       - minD is the minimum histogram distance that stops this"
   print "         simulation"
   exit()


totalSteps = int(sys.argv[1])    
timeSteps  = sys.argv[2]         
runSteps   = sys.argv[3]         
numLayers  = int(sys.argv[4])    
dT         = float(sys.argv[5])  
minD       = float(sys.argv[6])  


print '\ntotalSteps = %d timeSteps = %s runSteps = %s numLayers = %d dT = %f minD = %f \n' \
     % (totalSteps,timeSteps,runSteps,numLayers,dT, minD)


if totalSteps == 0:
   histD = 1000.0
else:
   read_histogram(totalSteps)
   read_rate(numLayers)
   read_small_large_bins()
   histD = read_histD()

while histD > minD and totalSteps < int(timeSteps):
    
    print '\ntotalSteps = %d \n' % totalSteps
    if totalSteps == 0:
       modify_input(0)
    else:
       modify_input(1)

    #time.sleep(10)
    cmd = path + '/Debug/stdp -n ' + runSteps + ' -p ' + path + '/input/params.stdp'
    #print cmd
    os.system(cmd)
    totalSteps = totalSteps + int(runSteps)

    # compute, write, and plot histogram
    histD = compute_histogram(totalSteps, int(runSteps), dT)

    # compute rate: arguments are: time, timeSteps, rateSteps, dT
    rate = compute_rate(numLayers,totalSteps*dT,int(runSteps), int(runSteps),dT)
    print '\n average rate: ' 
    for i in range(numLayers):
       print '%f ' % rate[i]
    print '\n'

    #if abs(rate-old_rate) < 0.1:
    # remove files
    if 0:
       cmd = 'rm ' + path + '/output/images/*'
       os.system(cmd)

       cmd = 'rm ' + path + '/output/*.pvp'
       os.system(cmd)

    old_rate = rate

#time.sleep(10)
print 'pass control to plot'
plt.show()
