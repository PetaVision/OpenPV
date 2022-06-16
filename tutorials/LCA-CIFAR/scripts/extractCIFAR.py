#! /usr/bin/env python

import os
import sys
import extractImages

progpath = sys.argv[0]
print(progpath)
progdir = os.path.dirname(progpath)
if len(progdir) == 0:
  progdir = '.'
cifarPath = progdir + '/' + '../cifar-10-batches-mat'
cifarPath = os.path.realpath(cifarPath)
mixed_file_pathname = cifarPath + '/' + 'mixed_cifar.txt'

if os.path.exists(mixed_file_pathname):
  print('WARNING: {} exists and is being deleted to prevent duplicate entries.'.format(mixed_file_pathname));
  os.remove(mixed_file_pathname) 

print('Extracting images..')

extractImages.extractImages(cifarPath + '/' + 'data_batch_1.mat', 1)
extractImages.extractImages(cifarPath + '/' + 'data_batch_2.mat', 2)
extractImages.extractImages(cifarPath + '/' + 'data_batch_3.mat', 3)
extractImages.extractImages(cifarPath + '/' + 'data_batch_4.mat', 4)
extractImages.extractImages(cifarPath + '/' + 'data_batch_5.mat', 5)
extractImages.extractImages(cifarPath + '/' + 'test_batch.mat', 0, False)
