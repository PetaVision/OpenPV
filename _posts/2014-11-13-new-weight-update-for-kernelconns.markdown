---
layout: post
title:  "New weight update for kernelconns"
date:   2014-11-13 23:01:55
author: Sheng Lundquist
categories: jekyll update
---

Hi all,

I'm committing a change to the way we reduce kernels. Specifically, reduce kernels now will average all kernels based on each pre/post for ONLY active pre AND post neurons. The difference between this change and what we've had before is specifically shrunken patches. Any nonexistent weight from shrunken patches previously was considered as a "0" when we reduce kernels, making our kernel reduction susceptible to edge effects, even when mirrorBC was on (weights do not look at extended post). Now, we properly account for shrunken patches, so that within a kernel, only actual existing weights (any weight connection attached to restricted post) is only being counted.

What does this mean for you? Probably not much. The change itself should reduce edge effects, which might be an improvement if you have really small input. DWMax's scale should mostly stay the same, with the exception that the scale may be minimally higher than previous dwMax.

A new feature that came out of this change (and why I made this change in the first place) is the fact that we can now mask the post layer in a connection to learn only off non-zero mask values. This means we can now specify a "Do not care" region in the post layer with respect to weight updates. The new HyPerConn parameters are:

*useMask*: specifies if the plastic connection uses a post mask for weight updates

*maskLayerName*: The binary (doesn't have to be binary, but 0 means do not learn here and non-zero means learn here) layer that specifies the post mask. Can be the post layer itself.

