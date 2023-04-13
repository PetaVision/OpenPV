import imageio
import numpy as np
import os
import pickle
import sys
import tarfile
import tempfile


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


def extract_images(input_path, output_dir, lead_digit, append_mixed=True):
  '''
  Extracts images from the given .mat file and writes them as .png files.
  input_path:   data file given as string, e.g 'data_batch_1'
  output_dir:   directory to write the results in. If does not exist it will be
                created. If it does exist, there is no checking for whether
                files exist or whether any previously existing files will be
                clobbered.
  lead_digit:   most significant digit in unique file number. Accounts for having 
                individual data files. lead_digit = 3 will result in numbers 3xxxx
  append_mixed: Whether to append the results to the mixed_cifar.txt file
  '''
  input_basename = os.path.basename(input_path)
  input_dirname = os.path.dirname(input_path)
  if len(input_dirname) == 0:
    input_dirname = '.'
  if input_basename.find('data_batch_') < 0 and input_basename.find('test_batch') < 0:
    print('filename does not begin with either "data_batch_" or "test_batch"', file=sys.stderr)
    exit(1)

  mkdir_if_needed(output_dir)
  
  contents = unpickle(input_path)
  num_ims, im_size = contents[b'data'].shape
  if im_size != 3072:
    print(f'data variable in {input_basename} does not have 3072 columns, as expected for CIFAR-10 data')

  # specify dimensions of image
  xdim = 32
  ydim = 32
  coldim = 3

  # get labels used to create subfolders for each label
  uniquelabels = np.unique(contents[b'labels'])

  # create subfolders
  for i in uniquelabels:
    subdir = output_dir + os.sep + str(i)
    mkdir_if_needed(subdir)

  # create randorder file for PV. Order within data file is random already
  # appends new lines to the end of the file
  randorder_pathname = output_dir + os.sep + input_basename + '_randorder.txt'

  batch_file = open(randorder_pathname, 'w')
  if append_mixed:
    mixed_file_pathname = output_dir + os.sep + 'mixed_cifar.txt'
    mixed_file = open(mixed_file_pathname, 'a')

  for i in range(num_ims):
    # get image path
    index = i + lead_digit * 10000
    label = contents[b'labels'][i]
    cifar_name = f'{output_dir}{os.sep}{label}{os.sep}CIFAR_{index:05d}.png'
    
    # process and save image
    im = contents[b'data'][i]
    im = im.reshape(coldim, ydim, xdim).transpose(1,2,0)
    imageio.imwrite(cifar_name, im)

    # write to batch txt files
    print(cifar_name, file=batch_file)
    if append_mixed:
        print(cifar_name, file=mixed_file)

  batch_file.close()
  if append_mixed:
    mixed_file.close()
  print(f'Finished {input_basename}')


def extract_cifar(inputfilename:str):
  print('Extracting images..')

  extractdir = tempfile.TemporaryDirectory()

  untar(inputfilename, extractdir.name)
  input_dir = os.path.join(extractdir.name, 'cifar-10-batches-py')
  output_dir = 'cifar-10-images'

  extract_images(os.path.join(input_dir, 'data_batch_1'), output_dir, 1)
  extract_images(os.path.join(input_dir, 'data_batch_2'), output_dir, 2)
  extract_images(os.path.join(input_dir, 'data_batch_3'), output_dir, 3)
  extract_images(os.path.join(input_dir, 'data_batch_4'), output_dir, 4)
  extract_images(os.path.join(input_dir, 'data_batch_5'), output_dir, 5)
  extract_images(os.path.join(input_dir, 'test_batch'), output_dir, 0, False)
  extractdir.cleanup()

def unpickle(path):
  with open(path, 'rb') as fileobj:
    contents = pickle.load(fileobj, encoding='bytes')
  return contents

def untar(filename:str, dirname:str="."):
  tar = tarfile.open(filename)
  tar.extractall(path=dirname)
  tar.close()

if __name__ == '__main__':
  if len(sys.argv) > 2:
      print(f'Usage: {sys.argv[0]} [filename]', file=sys.stderr)
      print(f'filename is the cifar-10-python.tar.gz file from the CIFAR-10 website', file=sys.stderr)
      print(f'Default filename is cifar-10-python.tar.gz', file=sys.stderr)
      exit(1)
  elif len(sys.argv) == 2 and (sys.argv[1] == "--help" or sys.argv[1] == "-h"):
      print(f'Usage: {sys.argv[0]} [filename]')
      print(f'filename is the cifar-10-python.tar.gz file from the CIFAR-10 website')
      print(f'Default filename is cifar-10-python.tar.gz')
      exit(0)
  if len(sys.argv) < 2:
    inputfilename = "cifar-10-python.tar.gz"
  else:
    inputfilename = sys.argv[1]

  extract_cifar(inputfilename)
