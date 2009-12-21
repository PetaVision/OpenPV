#!/usr/bin/python

# script to estimate the firing rate in L1 versus the background rate in retina
# copy this script to the location of your sandbox (STDP, marian, etc)

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

if len(sys.argv) < 6:
   print "usage: python transient.py totalSteps timeSteps writeSteps dT minD"
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
def read_rate():

   # initialize rate subplot
    plt.figure(1)
    plt.subplot(2,2,4)

    # read rate evolution 
    input = open(path + 'output/rate.stdp','r')

    while 1:
       line = input.readline()
       if line == '':break
       data = string.split(line,' ')
       time = eval(data[0])
       rate = eval(data[1])
       plt.plot(time/1000,rate,'o',color='b')
       
    input.close()

    plt.xlabel('TIME')
    plt.ylabel('AVERAGE RATE')
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
    plt.xlabel('Weight Bins')
    plt.ylabel('Count')
    plt.title('Weight Histogram')
    plt.xlim(0, 256)
    plt.grid(True)
    plt.draw()

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
          dh[k] = (1.0*abs((h[k]-h_old[k])))/max(h[k],h_old[k])
          k+=1   
       input.close()
       histD = max(dh)     # histogram distance
       print 'histD = %f\n' % (histD)
       output.write(str(time) + ' ' + str(histD) + '\n')
       output.close()

       plt.plot(np.arange(len(h_old)), h_old, 'o', color='r')
       plt.draw()

       #bx = fig.add_subplot(122, axisbg='darkslategray')
       plt.subplot(2,2,2)
       plt.plot(np.arange(len(dh)), dh, 'o', color='y')
       plt.xlabel('Weight Bins')
       plt.ylabel('Relative Bin Distance')
       plt.title('Weight Histogram Bin Distance')
       plt.xlim(0, 256)
       plt.grid(True)
       plt.hold(False)
       plt.draw()

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

totalSteps = int(sys.argv[1])    # length of previous simulation
timeSteps = sys.argv[2]     # length of simulation (timeSteps) from the beginning of 
                            # simulation
writeSteps = sys.argv[3]    # estimate histogram every writeSteps
dT         = float(sys.argv[4])    # simulation time step
minD      = float(sys.argv[5])         # stoping condition
#start_flag = int(sys.argv[5])    # starting flag

print '\ntotalSteps = %d timeSteps = %s writeSteps = %s dT = %f minD = %f \n' \
     % (totalSteps,timeSteps,writeSteps,dT, minD)


if totalSteps == 0:
   histD = 1000.0
else:
   read_histogram(totalSteps)
   read_rate()
   histD = read_histD()

while histD > minD and totalSteps < int(timeSteps):
    
    print '\ntotalSteps = %d \n' % totalSteps
    if totalSteps == 0:
       modify_input(0)
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
