#
# FFMPEG was set up in BASE
#

#
# GDAL 1.7 was set up in BASE
#

#
# libsvm
#
wget https://github.com/cjlin1/libsvm/archive/master.zip
unzip master.zip
cd libsvm-master
make -j4
make -j4 lib
cp libsvm.so.2 /usr/local/lib
ln -s /usr/local/lib/libsvm.so.2 /usr/local/lib/libsvm.so
cp svm.h /usr/local/include/
cp svm-scale svm-train svm-predict /usr/local/bin 
cd ..
rm -f master.zip

#
# liblinear
#
# tar xzvf liblinear-1.94.tar.gz
# cd liblinear-1.94
wget https://github.com/cjlin1/liblinear/archive/master.zip
unzip master.zip
cd liblinear-master
make -j4
make -j4 lib
cp liblinear.so.2 /usr/local/lib
ln -s /usr/local/lib/liblinear.so.2 /usr/local/lib/liblinear.so
ln -s /usr/local/lib/liblinear.so.2 /usr/local/lib/liblinear.so.1
cp linear.h /usr/local/include/liblinear.h
cp train /usr/local/bin/liblinear-train
cp predict /usr/local/bin/liblinear-predict
cd ..
rm -f master.zip

#
# MTRAND random number generator
wget http://www.bedaux.net/mtrand/mtrand.zip
unzip mtrand.zip
cd mtrand/
cp mtrand.h /usr/local/include
cd ..

#
# setup environment
#

# added to .bashrc
"""
##
## VAST
## 
export USER=/home/brumby
export VAST=$USER/Desktop/VAST
export VAST_DEV=$VAST/dev
export TMPDIR=$USER/tmp
export PYTHONPATH=$VAST_DEV/bin:$VAST/bin:$PYTHONPATH
export PANN_SRC=$VAST/src/pann3
## PARALLELISM
export MPIDIR=/usr/lib64/openmpi

## backwards compatibility
export NEURAL_TOOLS=$VAST
export NEURAL_DEV=$VAST_DEV

export PATH=$VAST_DEV/bin:$VAST/bin:$MPIDIR/bin:$PATH

export CPLUS_INCLUDE_PATH=$VAST_DEV/include:$VAST/include:/usr/include/openmpi-x86_64/:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=$CPLUS_INCLUDE_PATH

export VAST_LIB=$VAST_DEV/lib:$VAST/lib:$MPIDIR/lib

export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib64/atlas:$VAST_LIB:$LD_LIBRARY_PATH
export LIBRARY_PATH=$LD_LIBRARY_PATH

 """

#
# install third party packages
#

mkdir $USER/tmp

cd $USER/Desktop
mkdir VAST
cd VAST
mkdir src
mkdir dev
mkdir data
mkdir results

#
# set up LANL VAST code
#

#
# drag pann3 and pann_extras to $VAST
#

cd $VAST/src/pann_extras 
mv data/ $VAST/.
mv results $VAST/.
mv src/scripts $VAST/src
mv src/test $VAST/src
cd ..

# 
# test ATLAS on CentOS
# 
cd $VAST/src/pann_test
gcc -Wall -O5 test_cblas.c -o test_cblas -L/usr/lib64/atlas/ -lcblas -lm
./test_cblas
g++ -Wall -O5 test_cblas.cpp -o test_cblas_cpp -L/usr/lib64/atlas/ -lcblas -lm
./test_cblas_cpp
g++ -Wall -O5 test_cblas_clapack.cpp -o test_cblas_clapack_cpp -L/usr/lib64/atlas/ -lcblas -lclapack -lm -lrt
./test_cblas_clapack_cpp

#
# build VAST
# 

cd $PANN_SRC
make pann2
./pann2 --v2_off --color --image=example_kangaroo.jpg
./pann2 --v2_off --color --video=example_webcam.mp4 2>cerr.log



# success! (snap-shotted)
 
#!/usr/bin/python

##
## main

import os
import glob
import math
import re 
import operator
import random

import libvast

#
# data preprocessing 

#
# download tarballs to $VAST/data/imagenet

#
# can unpack tarballs in a directory using libvast.unpack_synsets()

#
# resize imagenet images to size shortedge = 256 pixels 

libvast.resize_all(in_path=os.environ.get('VAST')+'/data/imagenet/',out_path=os.environ.get('VAST')+'/data/imagenet_256/',size=256,subset=False,suffix='JPEG')

#
# set up trainig and validation set

categories = glob.glob(os.environ.get('VAST')+'/data/imagenet_256/*')
for c in range(len(categories)):
  categories[c] = categories[c].split('/')[-1]

print categories

run_name = 'demo_quick_01'

print "set up index files"
libvast.index_n_sets(run_name + '_index', 50, 0, 20, data_path=os.environ.get('VAST')+'/data/imagenet_256/*' )

## learn dictionary
## 
## V1
##
model_spec = " --r_scaling=1 --c_scaling=1 --s1_rfwidth=7 --s1_scaling=2 --s1_learn --s_random_features --s1_learn_n_prototypes=256 --s1_imprint_n_samples=512 --c1_rfwidth=1 --v2_off --v4_off --gm --gm_algorithm=ncbpdn --gm_lambda=0.03 --gm_rho=10.0 --gm_max_iterations=30 --gm_eta=0.03 "
libvast.learn_dictionary(run_name, model_spec, run_name+'_index_all_train.txt', log_file='cerr_train_s1_log.txt'  )

# ##
# ## V2
# ##
# model_spec = " --r_scaling=1 --c_scaling=2 --s1_rfwidth=7  --s1_scaling=2 --s1_apply=test04_s1_column.csv --c1_rfwidth=5  --v2_on --s2_rfwidth=5 --s2_scaling=2 --s2_learn --s_random_features --s2_learn_n_prototypes=512 --s2_imprint_n_samples=1024  --c2_rfwidth=3 --v4_off --gm --gm_algorithm=ncbpdn --gm_lambda=0.03 --gm_rho=10.0 --gm_max_iterations=10 --gm_eta=0.03 "
# libvast.learn_dictionary(run_name, model_spec, run_name+'_index_all_train.txt', log_file='cerr_train_s2_log.txt' )

## learn classifier model
##
model_spec = " --r_scaling=1 --c_scaling=2 --s1_rfwidth=7 --s1_scaling=2 --s1_apply="+run_name+"_s1_column.csv --c1_rfwidth=5 --v2_off --v4_off --gm_algorithm=simple_feed_forward "
print categories
libvast.train_model(run_name, model_spec, categories, index_prefix=run_name+'_index_', log_file="cerr_model_log.txt" )
