import os
import imageio
import scipy.io
import numpy as np
import sys


def mkdir_if_needed(path):
  '''
  If no file exists at the given path, creates a directory
  If a directory exists at the given path, does nothing
  If a non-directory file exists at the given path, exits with an error
  '''
  if not os.path.exists(path):
    os.mkdir(path)
  elif not os.path.isdir(path):
    print(f'Path "{path}" exists but is not a directory', file=sys.stderr)
    exit(1)


def extract_images(path, cnt, append_mixed=True):
  '''
  Extracts images from the given .mat file and writes them as .png files.
  path:         mat file given as string, e.g 'data_batch_1.mat'
  cnt:          most significant digit in unique file number. Accounts for having 
                individual .mat files. cnt = 3 will result in numbers 3xxxx
  append_mixed: Whether to append the results to the mixed_cifar.txt file
  '''
  basename = os.path.basename(path)
  dirname = os.path.dirname(path)
  if len(dirname) == 0:
    dirname = '.'
  if basename.find('data_batch_') < 0 and basename.find('test_batch') < 0:
    print('filename does not begin with either "data_batch_" or "test_batch"', file=sys.stderr)
    exit(1)
  if basename[-4:] != '.mat':
    print('filename does not have the extension ".mat"', file=sys.stderr)
    exit(1)

  output_dir_base = basename[:len(basename)-4]
  output_dir = dirname + os.sep + output_dir_base
  mkdir_if_needed(output_dir)
  
  matcontents = scipy.io.loadmat(path)
  num_ims, im_size = matcontents['data'].shape
  if im_size != 3072:
    print('data variable in {} does not have 3072 columns, as expected for CIFAR-10 data'.format(basename))

  # specify dimensions of image
  xdim = 32
  ydim = 32
  coldim = 3

  # get labels used to create subfolders for each label
  uniquelabels = np.unique(matcontents['labels'])

  # create subfolders
  for i in uniquelabels:
    subdir = output_dir + os.sep + str(i)
    mkdir_if_needed(subdir)
    # delete files remaining from a previous extractImages run, if any
    for file in os.listdir(subdir):
      os.remove(subdir + os.sep + file)

  # create randorder file for PV. Order within .mat file is random already
  # appends new lines to the end of the file
  randorder_pathname = output_dir + os.sep + output_dir_base + '_randorder.txt'
  mixed_file_pathname = dirname + os.sep + 'mixed_cifar.txt'

  batch_file = open(randorder_pathname, 'w')
  mixed_file = open(mixed_file_pathname, 'a')

  for i in range(num_ims):
    # get image path
    index = i + cnt * 10000
    label = matcontents['labels'][i][0]
    cifar_name = f'{output_dir}{os.sep}{label}{os.sep}CIFAR_{index:05d}.png'
    
    # process and save image
    im = matcontents['data'][i]
    im = im.reshape(coldim, ydim, xdim).transpose(1,2,0)
    imageio.imwrite(cifar_name, im)

    # write to batch txt files
    print(cifar_name, file=batch_file)
    if append_mixed:
        print(cifar_name, file=mixed_file)

  batch_file.close()
  mixed_file.close()
  print(f'Finished {output_dir_base}')


def extract_cifar():
  progpath = sys.argv[0]
  progdir = os.path.dirname(progpath)
  if len(progdir) == 0:
    progdir = '.'
  cifarPath = progdir + os.sep + '../cifar-10-batches-mat'
  cifarPath = os.path.realpath(cifarPath)
  mixed_file_pathname = cifarPath + os.sep + 'mixed_cifar.txt'

  if os.path.exists(mixed_file_pathname):
    print(f'WARNING: {mixed_file_pathname} exists and is being deleted to prevent duplicate entries.')
    os.remove(mixed_file_pathname) 

  print('Extracting images..')

  extract_images(cifarPath + os.sep + 'data_batch_1.mat', 1)
  extract_images(cifarPath + os.sep + 'data_batch_2.mat', 2)
  extract_images(cifarPath + os.sep + 'data_batch_3.mat', 3)
  extract_images(cifarPath + os.sep + 'data_batch_4.mat', 4)
  extract_images(cifarPath + os.sep + 'data_batch_5.mat', 5)
  extract_images(cifarPath + os.sep + 'test_batch.mat', 0, False)


if __name__ == '__main__':
  extract_cifar()