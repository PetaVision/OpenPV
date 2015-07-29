---
layout: post
title:  "Optimizations to PV"
date:   2014-11-7 23:01:55
author: Sheng Lundquist
categories: jekyll update
---

Hi all,

I've recently pushed another batch of important optimization updates to PV. Here's a quick summary.

WriteSparseActivity is now deprecated to support a new parameter, sparseLayer. Right now, the parameter is fully backwards compatible, but the parameter is expanded to consider the entire layer as sparse when doing receive from pre. What optimizations do you get when you set this flag you might ask? As sent in a previous email, the CPU recv from pre ("push" only active pre datastore elements to post gSyn) is now only looping over active indices IF AND ONLY IF sparseLayer (or writeSparseActivity) is set. Preliminary testing cut the time in recv from pre by about half for a 1% sparse layer. The functionality of writeSparseActivity (writing a sparse output pvp file) is still there with the new variable name.

Sparse convolutions are now on the GPU. The speedup here is not substantial, since the problem itself is not well suited for massive parallelism. That being said, the code is there and is working. Simply set updateGSynFromPostPerspective to false and receiveGpu to true. Preliminary testing says that sparse convolutions on the gpu is comparable with approximately 4 to 8 openmp threads with the new CPU sparse upgrade. Use this if you are using concurrency between layers to achieve load balancing between the CPU and the GPU.

Finally, dense convolutions for the one-to-many case on CUDNN is now implemented. The speedup here is substantial, about the same as the speedup achieved from dense convolutions for the many-to-one or one-to-one case. Use this in the same way you've done CUDNN before, making sure updateGSynFromPostPerspective is set to true and receiveGpu is set to true. Note that this dense convolution does not take advantage of sparseness.

Let me know if you have any problems with these optimizations, and any additional timing data would be greatly appreciated.


Sheng Lundquist
11/5/14

One more addition. Weight updates are now threaded. Here's some test numbers that I tried out with a simple weight update test:

1 thread: 17742 ms
2 threads: 9047 ms
4 threads: 5500 ms
8 threads: 3645 ms

This should help the runs where you are updating the weights every timestep.


Sheng Lundquist
11/7/14

PetaVision defaults to using device 0, and the timing info in L1's receive and update on the GPUs says it's running. It would be interesting to see a comparison between the sparse convolution on the gpu vs cpu with Error's receive.

Pete had an interesting thought the other day. We're not triggering our error layers because it needs to be updated every time-step since V1 (L1 in this run's case) needs to be updated. However, Error's receive includes the receive from the input layer as well. What we may be seeing in Error's timing info may be the Image to Error receive as opposed to V1 to Error. We had a conversation about maybe triggering individual receives, but a temporary fix is to put the Image to Error IdentConn on GPUs. (you may have to reverse the IdentConn with a channel code of -1 and transpose that connection to do RecvFromPost CUDNN convolution).

One final thought is that I remember Dylan mentioned that you had multiple GPU cards in your machine. If you split the MPI up to 2 and use the flag *-d -1*, you should be able to split your MPI runs onto each GPU.

{% highlight bash %}
mpirun -np 2 Release/DepthLCA -p input/myParam.params -t 4 -d -1
{% endhighlight %}

