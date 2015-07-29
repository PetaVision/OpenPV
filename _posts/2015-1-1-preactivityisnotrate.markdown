---
layout: post
title:  "preActivityIsNotRate, activityIsSpiking, and convertRateToSpikeCount"
date:   2015-1-1 23:01:55
author: Pete Schultz
categories: jekyll update
---

Happy New Year, everybody!  Based on the recent discussion, I got rid of the HyPerConn parameter preActivityIsNotRate, the all-time worst-named parameter.  It is currently an error to set preActivityIsNotRate to true in the params file.  If preActivityIsNotRate is set to false, there is a warning but the run continues.

Instead, HyPerConn has a parameter named convertRateToSpikeCount, which defaults to false.  If that flag is false, the connection does not scale the presynaptic activity before computing GSyn.  This is the same as before (which is why it makes sense for it to be a warning but not an error if you set preActivityIsNotRate to false).  However, if the flag is true, the connection checks whether the pre-layer is spiking, and only scales the presynaptic activity if the layer is not spiking.

If it does rescale, the rescaling factor is what was there before: if the time constant for the post channel is positive, it uses (1-exp(-dt/tau))/exp(-dt/tau); otherwise it uses dt.

How does the connection know whether the presynaptic layer is spiking?  HyPerLayer now has a pure virtual method activityIsSpiking().  Because it's pure (the declaration in HyPerLayer.hpp is "virtual bool activityIsSpiking() = 0;"), subclasses of HyPerLayer must implement activityIsSpiking().  I did that so that it would be necessary to explicitly decide whether any new HyPerLayer subclass is spiking or not.  However, a subclass of a subclass of HyPerLayer will inherit activityIsSpiking() unless it overrides.

activityIsSpiking() is not usually determined by a params file parameter.  The implementation of the layer's updateState determines whether the layer is spiking or not.  Retina is an exception because it already has a spikingFlag parameter, and Retina::updateState produces spikes or continuous output based on the value of that flag.  Right now, LIF and it's subclasses BIDSLayer and LIFGap have activityIsSpiking() return true; Retina returns the value of spikingFlag, and all other subclasses in trunk or PVSystemTests return false.  Please let me know if there are any other spiking subclasses.

If your sandbox or local copy defines a subclass of HyPerLayer, that subclass will need to implement activityIsSpiking().

With these recent commits, all the system tests pass.  However, there are no system tests yet that set convertRateToSpikeCount to true, and StochasticReleaseTest sets dt to 1.  We should add system tests to check these features soon.

