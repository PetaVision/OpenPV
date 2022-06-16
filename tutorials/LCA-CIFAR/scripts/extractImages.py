import os
import imageio
import scipy
import scipy.io
import sys

# If no file exists at the given path, creates a directory
# If a directory exists at the given path, does nothing
# If a non-directory file exists at the given path, exits with an error
def mkdir_if_needed(path):
  if not os.path.exists(path):
    os.mkdir(path)
  elif not os.path.isdir(path):
    print('Path "{}" exists but is not a directory'.format(basename), file=sys.stderr)
    exit(1)

# Extracts images from the given .mat file and writes them as .png files.
# path:         mat file given as string, e.g 'data_batch_1.mat'
# cnt:          most significant digit in unique file number. Accounts for having 
#               individual .mat files. cnt = 3 will result in numbers 3xxxx
# append_mixed: Whether to append the results to the mixed_cifar.txt file
def extractImages(path, cnt, append_mixed=True):
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
  output_dir = dirname + '/' + output_dir_base
  mkdir_if_needed(output_dir)
  
  matcontents = scipy.io.loadmat(path)
  xl, yl = matcontents['data'].shape
  if yl != 3072:
    print('data variable in {} does not have 3072 columns, as expected for CIFAR-10 data'.format(basename))

                                # initialize image matrix to CIFAR image dimensions
  xdim = 32
  ydim = 32
  coldim = 3
  im = scipy.zeros( (xdim, ydim, coldim), dtype=scipy.uint8)
                                # get labels used to create subfolders for each label
  uniquelabels = scipy.unique(matcontents['labels'])
  mi = min(min(matcontents['labels']))
  ma = max(max(matcontents['labels']))

                                # create subfolders
  for i in uniquelabels:
    subdir = output_dir + '/' + str(i)
    mkdir_if_needed(subdir)
    # delete files remaining from a previous extractImages run, if any
    for file in os.listdir(subdir):
      os.remove(subdir + '/' + file)

                                # create randorder file for PV. Order within .mat file is random already
                                # appends new lines to the end of the file
  randorder_pathname = output_dir + '/' + output_dir_base + '_randorder.txt'
  mixed_file_pathname = dirname + '/' + 'mixed_cifar.txt'

  batch_file = open(randorder_pathname, 'w')
  mixed_file = open(mixed_file_pathname, 'a')

  for i in range(xl):
    index = i + cnt * 10000;
    CIFAR_name = '{0}/{1}/CIFAR_{2:05d}.png'.format(output_dir, matcontents['labels'][i][0], index)
    for k in range(yl):
      xi = k % xdim
      yi = ((k-xi)//xdim) % ydim
      ci = ((k-xi-xdim*yi)//(xdim*ydim)) % coldim
      im[xi,yi,ci] = matcontents['data'][i, k]

    imageio.imwrite(CIFAR_name, im)
    print(CIFAR_name, file=batch_file)
    if append_mixed:
        print(CIFAR_name, file=mixed_file)

  batch_file.close()
  mixed_file.close()
  print('Finished {}'.format(output_dir_base))
