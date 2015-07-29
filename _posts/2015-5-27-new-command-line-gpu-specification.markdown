---
layout: post
title:  "New Command Line GPU Specification"
date:   2015-5-27 23:01:55
author: Sheng Lundquist
categories: jekyll update
---

Hi all,

I've finally made a change to do specific mapping of gpus to specific mpi processes. First off, the -d -1 case is no longer supported; rather, PetaVision automatically does automatic mapping of GPUs by default.

The -d flag now takes arguments as such:

{% highlight bash %}
mpirun -np 4 Debug/BasicSystemTest -t 1 -d  #Will default to automatic mapping
mpirun -np 4 Debug/BasicSystemTest  -t 1 -d 0  #All mpi processes will use GPU 0
mpirun -np 4 Debug/BasicSystemTest -t 1 -d 3,2,1,0  #mpi process 0 will use GPU 3, mpi process 1 will use GPU 2, etc.
{% endhighlight %}

Additionally,

{% highlight bash %}
mpirun -np 4 Debug/BasicSystemTest -t 1 -d 0,1,2 #Will fail, # of specified GPUs must be >= number of mpi processes
mpirun -np 4 Debug/BasicSystemTest -t 1 -d 0,1,2,3,4,5,6,7 #Will silently throw away 4,5,6,7 and only use 0,1,2,3 for GPU mapping.
{% endhighlight %}

