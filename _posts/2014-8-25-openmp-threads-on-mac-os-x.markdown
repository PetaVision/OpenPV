---
layout: post
title:  "OpenMP Threads on Mac OSX"
date:   2014-08-25 23:01:55
author: Pete Schultz
categories: jekyll update
---

Hi everyone.  I figured out how to compile petavision on macs using the *OPEN_MP_THREADS* option.  The problem is that homebrew's default compilers are clang and clang++, so that mpicc and mpic++ get wrapped around those compilers, but clang and clang++ don't handle the omp stuff yet.

The solution is to wrap openmpi around gcc and g++; but a secondary problem is that in OS X Mavericks, apparently, gcc and g++ are actually clang and clang++ in disguise.  But gcc is available through homebrew, so we can get the compilers that understand omp, and then build OpenMPI using those compilers.  Here's what I did, starting with a fresh installation of xcode and homebrew.

{% highlight bash %}
brew install gcc
# the above installs /usr/local/bin/gcc-4.9 and /usr/local/bin/g++-4.9  There are still /usr/bin/gcc and /usr/bin/g++ that point to the Apple versions.
brew install --cc=gcc-4.9 openmpi
brew install --cc=gcc-4.9 gdal
brew install --cc=gcc-4.9 libsndfile
brew install --cc=gcc-4.9 cmake
{% endhighlight %}

The --cc option switches the compiler used.  It seems that an absolute path doesn't work as the argument, but gcc-4.9 does.  That changes both the C and C++ compilers.
I'm pretty sure that the cmake command doesn't require the compiler selection, and it probably isn't necessary for the gdal and libsndfile commands either.  But those are the commands I ran.

Then building PetaVision with *OPEN_MP_THREADS* set should work.  You can use *make VERBOSE=1* to check that it is compiling with options *-fopenmp* and *-DPV_USE_THREADS*

The commands *mpicc --showme* and *mpic++ --showme* show what commands the mpi compilers are wrappers for; you can verify that they call */usr/local/bin/{gcc,g++}* and not clang/clang++


Gerd Kunde
10/23/14

Dear All,

I am just trying to make PV run on 10.9.5 with XCODE 6.1, there are
a few differences to what Pete wrote below.

It appears that BREW knows about the gcc dependence :

{% highlight bash %}
brew install openmpi
==> Installing dependencies for open-mpi: gmp, mpfr, libmpc, isl, cloog, gcc, openssl, libev
==> Installing open-mpi dependency: gmp
==> Downloading https://downloads.sf.net/project/machomebrew/Bottles/gmp-6.0.0a.mavericks.bottle.ta
######################################################################## 100.0%
{% endhighlight %}

alas there is a warning a little later:

{% highlight text %}
GCC has been built with multilib support. Notably, OpenMP may not work:
https://gcc.gnu.org/bugzilla/show_bug.cgi?id=60670
If you need OpenMP support you may want to
brew reinstall gcc --without-multilib
{% endhighlight %}

This command takes forever ...

In addition we need to add to the instructions that after
installing XCODE and its command line extensions one has
to say: 

{% highlight bash %}
xcodebuild -license
{% endhighlight %}

otherwise homebrew will not install but say:

{% highlight bash %}
==> /usr/bin/sudo /usr/bin/xcode-select --install
xcode-select: note: install requested for command line developer tools
Press any key when the installation has completed.
==> Downloading and installing Homebrew...
Agreeing to the Xcode/iOS license requires admin privileges, please re-run as root via sudo.
{% endhighlight %}

but even that does not work until the *xcodebuild -license* has been completed.

