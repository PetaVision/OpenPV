---
layout: post
title:  "ImprintConn, new reduceKernels method and faster learning"
date:   2014-03-06 23:01:55
author: Sheng Lundquist
categories: jekyll update
---

Hi all,

I'm about to commit a change to PetaVision that changes the scale of dWMax in KernelConns. Previously, any dw was rescaled by dividing all dw's by numNeurons/numKernels. Unfortunately, this made the scale of dWMax dependent on the size of the image columns. We are now dividing dWMax by the number of features that are actually contributing (non-zero pre-synaptic activity) to the shared weights of KernelConns.

On a related note, I also implemented an ImprintConn. The only difference between ImprintConn and KernelConn is that ImprintConn keeps track of each dictionary element's last active time (updated on each weight update). There exists a parameter for ImprintConn called imprintTimeThresh, which essentially says if a dictionary element has not been active for the past imprintTimeThresh timesteps, that dictionary element will get imprinted from the post synaptic layer (in the case of LCA's, this will be the residual layer).

Please don't hesitate to bug me if you need any help or clarification with these changes.

Sheng
