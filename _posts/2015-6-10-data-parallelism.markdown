---
layout: post
title:  "Data Parallelism"
date:   2015-6-10 23:01:55
author: Sheng Lundquist
categories: jekyll update
---

Hi all,

TL;DR: New dimension in PV to implement data parallelism.

I think I have an idea of how to implement data parallelism into PetaVision. This email is also to just jot my ideas down so that i have a hard copy of it.

We implement a new dimension into PetaVision. I'll call it the "Batch" dimension as of now, but this can be changed.

Layers:
All layers contain a 4d matrix, from slowest to fastest [batch, ny, nx, nf].
Current restriction is that all layers in a run must contain the same batch size. We can either specify that batch size in the HyPerCol, or require the user to specify it for all layers individually, in case we want to implement a "many to many batch" method in the future.
Layer recv can either be either on CPU (looping over batches, we have CPU parallelism options, such as a new MPI split dimensions, although it might not play nice with GPUs), or on GPUs (CUDNN provides a 4th dimension that we're not currently using, which I'm sure i can find somehow to do extra-shared weights [see below] across all batches on the GPU). I'll have to put more thought into how we can actually do CPU MPI split + 1 GPU, although i'm sure we can have separate MPI's talk to 1 GPU in an organized manner to get all the data on there.

Connections:
Extra-Shared weights means it's now replicated across batches.
Shared weights means it's independent across batches, but shared weights as we have now
Non-shared means everything has an independent weight.
Arbors are pretty in place for what we can do for the latter 2.

The idea here is to build a network (say LCA) that have independent movies playing in each batch dimension. All layers in the network will have their separate layer batch dimensions, making it technically run batch number of independent LCA networks. All connections will be Extra-Shared, with a sync block to reduce *dw_weights* across all batches every weight update. One GPU can do the convolution across all the batches when V1 recvs, while we have CPU parallelism when error recvs.

In a network such as AlexNet, we convolve with the same weights across everything in the batch dimension, making what they call a "mini-batch" equivalent to our batch dimension. I think we'll get the data parallelism GPU speedup they get by using CUDNN.

Sheng


Garrett Kenyon
6/11/15

One problem, LCA is a dynamical model.  It doesn't make sense to update a dynamical variable in batch mode.  The membrane potential of a neuron is only defined for a single neuron, not for a batch of neurons.  The batch dimension needs to spread across machines or across MPI processes.  It shouldn't be a physical dimension in the way the other 3 are.


Sheng Lundquist
6/11/15

In LCA, I would say there are batch # sets of neurons, each with their own membrane potentials. The only thing shared across batches is dictionary weights. That way, each batch dimension of LCA achieves sparse encoding of different images with the same dictionary. Every weight update, we reduce all the dictionary updates.





