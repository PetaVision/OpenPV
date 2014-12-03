10/18/2013 brumby@lanl.gov

general user:
screen name: VAST user 
username: vast
password: v@st_u2er

set root password: 
http://wiki.centos.org/TipsAndTricks/ResetRootPassword
"""
Interrupt the boot at the GRUB stage and boot to runlevel 1, 
AKA single user mode. Interrupt GRUB by typing a character such as "space" then append to the kernel line by typing "a", 
backspacing through "rhgb quiet" and appending " 1<enter>". 
This will give you a root shell and not a login prompt. 
From there you can use the "passwd" command to set a new root password.

Other user passwords can be reset, and other administrative tasks can be performed as well. 
Exiting the single user root shell will boot to the multi-user mode defined in /etc/inittab.
"""
passwd
[enter the root password: v@st_r00t]
[re-enter the root password: v@st_r00t]
restart

#
# LANL only
yum proxy configuration: 
http://trac.lanl.gov/cgi-bin/ctn/trac.cgi/wiki/SelfHelpCenter/ProxyUsage
"""
To use yum through the proxy add this line to /etc/yum.conf
proxy=http://proxyout.lanl.gov:8080
"""

#
# update the system:
# 

su root
yum update -y

# <restart>

#
# add developer/debugger tools
#

#  appending -y to all of the lines below allows entire block to be run at once
# cut and paste following blocks into vm terminal using apple left click

# ******* BEGIN CUT AND PASTE *******************************
yum groupinstall "Development tools" "Debugging tools" -y
yum install openmpi openmpi-devel -y
yum install atlas atlas-devel blas blas-devel lapack lapack-devel -y
yum install emacs -y
yum install screen -y
yum install readline-devel ncurses-devel -y
yum install openssl-devel -y
yum install bzip2-devel -y
yum install tk -y 
yum install tk-devel -y 
yum install libpng -y 
yum install libpng-devel -y 
yum install libjpeg-turbo libjpeg-turbo-devel libtiff libtiff-devel -y
yum install cmake -y
yum install xterm -y
yum install gtk2-devel -y

#
# upgrade yum to draw packages from EPEL repository
# cut and paste following block into vm terminal
wget http://mirror-fpt-telecom.fpt.net/fedora/epel/6/i386/epel-release-6-8.noarch.rpm
rpm -ivh epel-release-6-8.noarch.rpm

yum install GraphicsMagick-devel -y
yum install dkms -y
yum install libvdpau -y
yum install protobuf-devel -y 
yum install leveldb-devel -y 
yum install snappy-devel -y 
yum install opencv-devel -y 
yum install boost-devel -y 
yum install hdf5-devel -y
#yum install cmake28
yum install npm -y # also builds node.js
yum install jq -y # jq - JSON commandline sed



#
# Google Chromium browser - built-in MP4 video support
# wget -e use_proxy=yes -e http_proxy=proxyout.lanl.gov:8080 http://people.centos.org/hughesjr/chromium/6/chromium-el6.repo
cd /etc/yum.repos.d/
wget http://people.centos.org/hughesjr/chromium/6/chromium-el6.repo
yum install chromium -y
cd

#
# youtube-dl
yum install youtube-dl -y

#
# unrar
# http://www.tecmint.com/how-to-open-extract-and-create-rar-files-in-linux/
wget http://pkgs.repoforge.org/rpmforge-release/rpmforge-release-0.5.2-2.el6.rf.x86_64.rpm
rpm -Uvh rpmforge-release-0.5.2-2.el6.rf.x86_64.rpm
yum install unrar -y

#
# GDAL 1.7
#
# updated: Oct 9 2014
# problem: yum'ing GDAL reported needed libgeotiff package
# solution: Google search for "yum repo libgeotiff x86_64"
# libgeotiff-devel available from: http://elgis.argeo.org/repos/6/
#rpm -Uvh elgis-release-6-6_0.noarch.rpm 
# lib armadillo from http://proj.badc.rl.ac.uk/cedaservices/attachment/ticket/670/armadillo-3.800.2-1.el6.x86_64.rpm
rpm -Uvh http://elgis.argeo.org/repos/6/elgis-release-6-6_0.noarch.rpm
wget http://proj.badc.rl.ac.uk/cedaservices/raw-attachment/ticket/670/armadillo-3.800.2-1.el6.x86_64.rpm
rpm -Uvh armadillo-3.800.2-1.el6.x86_64.rpm
yum install libgeotiff -y 
yum install libgeotiff-devel -y
yum install gdal -y 
yum install gdal-devel -y


#
# FFMPEG
#
# http://trac.ffmpeg.org/wiki/CompilationGuide/Centos

yum install yasm -y

# x264 - GPL!
git clone git://git.videolan.org/x264.git
cd x264
./configure --enable-shared --enable-static
make -j4
make install -j4
cd ..
rm -fr x264

# built ffmpeg from stable tarball, using:
# downloading takes a while on wireless...
git clone git://source.ffmpeg.org/ffmpeg.git ffmpeg
cd ffmpeg
./configure --prefix=/usr/local --enable-shared --enable-gpl --enable-libx264
make -j4
make install -j4
cd ..
rm -fr ffmpeg


#
# FFTW 3.3.4
#

export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib64/atlas/
module add openmpi-x86_64
wget ftp://ftp.fftw.org/pub/fftw/fftw-3.3.4.tar.gz
tar xzvf fftw-3.3.4.tar.gz
cd fftw-3.3.4
./configure --prefix=/usr/local --enable-shared --enable-threads --enable-openmp --enable-mpi --enable-sse2 
make -j4
make -j4 install
cd ..
rm -fr fftw-3.3.4



## put in separate shell script as a stand alone test
#
# optional: test FFTW installation
if 0
mkdir test
cd test
vi fftw_test.cpp
"""
#include <fftw3.h>
#include <iostream>

int main()
{
    std::cerr << "init\n" << std::flush;

    long N = 1024*1024*16;

    fftw_complex *in, *out;
    fftw_plan p;

    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

    std::cerr << "plan\n" << std::flush;
    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    std::cerr << "execute\n" << std::flush;
    fftw_execute(p);

    std::cerr << "clean-up\n" << std::flush;
    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);

    return 0;
}

// g++ -o fftw_test -I../../include -L../../lib -lfftw3 -lm fftw_test.cpp
// export LD_LIBRARY_PATH=../../lib/:${LD_LIBRARY_PATH}
// ./fftw_test
"""
cd ..

cd ..
rm -fr fftw-3.3.4
fi

## end test



#
# PYTHON

export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib64/atlas/
module add openmpi-x86_64

#
# install Python 2.7.7
# export LD_LIBRARY_PATH=/usr/local/lib
wget https://www.python.org/ftp/python/2.7/Python-2.7.tgz
tar xzvf Python-2.7.tgz
cd Python-2.7
./configure --enable-shared
make -j4 
make -j4 install
cd ..

# build distribute-0.6.35 --> 
wget https://bitbucket.org/pypa/setuptools/raw/bootstrap/ez_setup.py
python2.7 ez_setup.py
easy_install-2.7 pip

#
# numerical Python basic packages

pip2.7 install numpy --upgrade  # 1.8.1

#
## matplotlib needs freetype2 2.4
## I downloaded freetype-2.5.2.tar.gz from
## http://download.savannah.gnu.org/releases/freetype/freetype-2.5.2.tar.gz
wget http://download.savannah.gnu.org/releases/freetype/freetype-2.5.2.tar.gz
tar -xzvf freetype-2.5.2.tar.gz
cd freetype-2.5.2
./configure --enable-shared
make -j4 
make -j4 install
cd ..
#

pip2.7 install scipy  
pip2.7 install scikit-learn
pip2.7 install six --upgrade
pip2.7 install scikit-image
pip2.7 install pillow  # adds PIL
pip2.7 install matplotlib  # 1.3.1

#
# install parallel python
wget http://www.parallelpython.com/downloads/pp/pp-1.6.4.tar.gz
tar xzvf pp-1.6.4.tar.gz
cd pp-1.6.4
python2.7 setup.py install
cd ..


# ******* END CUT AND PASTE *******************************

#
# mercurial 3
wget http://mercurial.selenic.com/release/mercurial-3.1.2.tar.gz
tar -xzvf mercurial-3.1.2.tar.gz
cd mercurial-3.1.2
python2.7 setup.py install
cd ..

## do this as a cat to automate?
vi ~/.hgrc
"""
[ui]
username = Steven P Brumby <brumby@lanl.gov>
"""

#
# ANTIVIRUS

#
# antivirus for CentOS - ClamAV
yum install clamav clamd -y
#/etc/init.d/clamd on  ## gives error
chkconfig clamd on
/etc/init.d/clamd start
/usr/bin/freshclam
vi /etc/cron.daily/manual_clamscan
"""
#!/bin/bash
SCAN_DIR="/home"
LOG_FILE="/var/log/clamav/clamscan.log"
/usr/bin/clamscan -i -r $SCAN_DIR >> $LOG_FILE
"""
chmod +x /etc/cron.daily/manual_clamscan
/usr/bin/clamscan -r /home/brumby/Desktop/caffe/ > /var/log/clamav/manual_clamscan_all.log
# print just infected files
/usr/bin/clamscan -i -r /home/brumby/Desktop/caffe/ > /var/log/clamav/manual_clamscan_infected.log

exit root



## move to end since this step requires restart and is not needed by anything else
## skip the following for machine with non-nvidia GPU?
#
# get CUDA from NVIDIA

# repo from :  https://developer.nvidia.com/cuda-downloads?sid=550023

rpm --install cuda-repo-rhel6-*.rpm
yum install cuda -y
#
# warning: installing CUDA casues a reboot problem, due to updates to video driver caused by nvidia drivers
# http://fertoledo.wordpress.com/2013/08/30/linux-centos-system-stops-booting-at-atd-service/
# fix for this:
# boot in no graphics (runlevel 3) mode
cd /etc/X11
mv xorg.conf xorg.conf-old
# reboot



#
# set /etc/resolve.conf on AWS/EC2 instance
"""
# Generated by NetworkManager


# No nameservers found; try putting DNS servers into your
# ifcfg files in /etc/sysconfig/network-scripts like so:
#
nameserver 8.8.8.8
nameserver 8.8.4.4
# DOMAIN=lab.foo.com bar.foo.com
"""
