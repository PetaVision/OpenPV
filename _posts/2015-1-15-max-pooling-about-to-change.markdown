---
layout: post
title:  "Max Pooling about to change"
date:   2015-1-15 23:01:55
author: Garrett Kenyon
categories: jekyll update
---

hey,

I'm about to commit some fugly but necessary changes to how MAX pooling is treated by PetaVision.

the problem I was running into was that when MAX pooling over a deep layer, say S2, with say nf > 1K,  the old PetaVision method was convolving over the entire weight cube, which is nxp * nyp * nf, which becomes very large when pooling over a large region (12x12x1536 in my case).  The vast majority of the weights in the cube are zero, only weights connecting neurons with the same feature index are non-zero, and all of those are identically 1.  Yet the weight vector itself was over 20GB, just to store 1 scalar that of course I don't even need!  Anyway, to circumvent these issues, I've added IF statements to PetaVision that avoid ever allocating a weight vector if the accumulate type is max\_pooling or sum\_pooling and the receiveFromPostPerspective = false.  If the latter is true, we need to take a transpose, which requires the weights to exist.  Anyway, all of this is hopefully going to get cleaned up going forward but for the moment: if you are using MAX pooling, my commit will break your code unless you are max pooling from the post perspective.  To use max pooling in the future, you should set: weightInitType = "MaxPoolingWeight" and receiveFromPostPerspective = false.  

Hopefully, I haven't broken anything else.  all the system tests pass and my own complicated 2-layer sparse model with a layer of max pooling and a layer of sum pooling and an SLP backend, seems to be working.

