---
layout: post
title:  "GPU Update"
date:   2014-08-28 23:01:55
author: Sheng Lundquist
categories: jekyll update
---

Hi all,

After lots of work, GPUs are now up and running in PetaVision. Here's what you need to know to get your favorite run running on a GPU.

Here's what your directory structure should look like.

{% highlight text %}
Workspace
    PetaVision (trunk)
        PVSystemTests
{% endhighlight %}

#Setup:

First things first, there's many CMake updates, so clear all of your cache and cmake files first.

If you're using Cuda, add */usr/local/cuda-6.0/lib64* (or wherever your cuda library is) to your *LD_LIBRARY_PATH*, and add */usr/local/cuda-6.0/bin* to your *PATH*.

Copy *trunk/docs/cmake/CMakeLists.txt* into your Workspace directory.

From your workspace directory, run 

{% highlight bash %}
ccmake .
{% endhighlight %}

(Note: if you're on the servers, chances are you'll need to run *ccmake28*, since the ccmake command is an older version of cmake that will not work)

Configure by entering *c*. Don't worry if there's a few errors. First thing to change is *PV_DIR* to match your trunk directory. Change your *CMAKE_BUILD_TYPE* to either Debug or Release.

Next, decide if you're going to run using OpenCL or Cuda. If you have an Nvidia card, I suggest running Cuda. Otherwise, OpenCL should work on any video card you have.

##Cuda:
Set *CUDA_GPU* to true. If you would like to optimize your Cuda compiled code, set *CUDA_RELEASE* to true as well. From here, press *c* again to configure.

If cmake is complaining about CUDA paths it can't find, here's the common paths that are not found (press *t* for toggle advanced mode to find these options) :

{% highlight text %}
CUDA_CUDART_LIBRARY = /usr/local/cuda-6.0/lib64/libcudart.so
CUDA_CUDA_LIBRARY = /usr/lib64/libcuda.so
CUDA_TOOLKIT_INCLUDE = /usr/local/cuda-6.0/include
CUDA_TOOLKIT_ROOT_DIR = /usr/local/cuda-6.0
{% endhighlight%}

##OpenCL:
Set *OPEN_CL_GPU* to true. If you're running on a linux machine, set *OpenCL_dir* to your GPU driver (there should exist a directory *OpenCL_dir/include/CL/opencl.h*. Cuda drivers should have this as well). If you're running on a mac, you can leave this field as is, since it doesn't use this parameter.

From here, press *c* until the *g* (generate) option appears, and generate your makefile.

#Running:
As of right now, we only allow GPU receive synaptic input from TransposeConns. Future work will be done to remove this restriction. Find the connections that you would like to accelerate on the GPUs. Note that GPUs are very good at recvFromPost as opposed to recvFromPre. In other words, you'll want to put your Error to V1 recv on GPUs as opposed to V1 to Error.

There are a few parameters to take note in these connections:

*updateGSynFromPostPerspective* : This value should be set to true to recvFromPost.
*receiveGpu* : This is a flag to tell PetaVision that this is the connection to accelerate with GPUs. Set this to true.
*numXLocal*, *numYLocal*, *numFLocal*: Without getting into too much implementation details, for best results, set *numFLocal* to the number of post synaptic features. Set *numYLocal* to 1. Set *numXLocal* such that *numFLocal* \* *numXLocal* <= maximum threads group size. To find out that number, doing a run with GPU compilation should print out avaliable devices, and should print out that number. Furthermore, numXLocal must be divisible by the post layer size.

When running PetaVision, there is a flag *-d* that defines which device you would like to use. Doing a run with GPU compilation should give you what device numbers correspond to which GPUs. This value defaults to 0 if not used.

How do you know if the GPU is running? I suggest running GPUSystemTest in PVSystemTests:

{% highlight bash %}
Debug/GPUSystemTest -p input/cl_postTest.params -d 0
{% endhighlight %}

Check out the timing information. The GPU connection should be much faster in recvSynapticInput than the CPU implementation.

If you want even more speed and have multiple GPUs on a machine, you can combine MPI and GPU implementations. By setting the device (*-d*) to -1, each mpi process will grab it's rank's device. This combined with threads should give you the best performance. An example run command on Neuro (24 cores, 4 beefy teslas) can be as follows:

{% highlight bash %}
mpiexec -np 4 Release/myRun -p myParameters.params -d -1 -t 6
{% endhighlight %}

Let me know if you have any problems with compiling or running on GPUs.

Happy speedups.

