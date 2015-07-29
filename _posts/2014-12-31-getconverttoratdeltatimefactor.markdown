---
layout: post
title:  "getConvertToRateDeltaTimeFactor"
date:   2014-12-31 23:01:55
author: Garrett Kenyon
categories: jekyll update
---

the problem as I see it is the following:

1) stochastic output needs to be scaled by ~dt before the output activity is calculated, since the output needs to be converted into a probability.

2) all rate-coded (e.g. continuous or constant) input to any dynamical variable (a variable obeying a 1st order diff eq) needs to be scaled by ~dt but this scaling can occur at any point in the chain

3) spiking input should never be scaled (or, equivalently, always scaled by 1).


For example, a non-spiking input to an LCA layer, which is a dynamical layer, needs to be scaled by ~dt, but this scaling can occur at any point in the chain.  The most natural places to scale the input to an LCA layer (or to any dynamical layer), is in the update\_state equations, as that's where we expect to see multiplicative factor of the form: dt/tau.   However, sometimes the scaling has to occur long before the update\_state equations are called (i.e. if the input is stochastic) and sometimes the input shouldn't be scaled at all (i.e. if the input is spiking).  

I honestly don't see the resolution that addresses all the possible cases in an elegant and intuitive manner.


Pete Schultz 
12/31/14

Regarding this:
> 2) all rate-coded (e.g. continuous or constant) input to any dynamical
> variable (a variable obeying a 1st order diff eq) needs to be scaled by ~dt
> but this scaling can occur at any point in the chain

I'd quibble with the "any point in the chain" part.  If we scale by dt during the calculation of GSyn, we still have to scale everything else (V and A) in the diff eq  during updateState.  Further, the apparent interchangeability of rescaling in deliver() and rescaling in updateState() happens only because dV/dt depends linearly on GSyn.  If for some reason the equation was dV/dt=1/tau\*(GSyn^2 - V + A), then rescaling GSyn and (-V+A) individually would be wrong.

In case (1), the presynaptic activity A is a probability density: the relevant quantity is the probability that the event occurs in an interval [t, t+dt), which is is A\*dt.  Hence we rescale, and as Gar pointed out yesterday, we have to rescale before rolling the dice.

In case (2), the relevant quantity is A itself, not A\*dt.  We do *not* rescale the input GSyn.  The rescaling that HyPerLCALayer does in updateState is not modifying GSyn, it's modifying \Delta V.  This is the reason that, as Gar observed, it's most natural to rescale in updateState.

In case (3), the spiking case, the relevant quantity is not actually A, but A times a Dirac delta function.

The difficulties arise when we mix cases.  The problems we ran into in Innovation House, if I remember correctly, were caused when the presynaptic layer was generating case-(2) output, but the post-synaptic layer was expecting case-(3) input.  We interpreted the output of the presynaptic layer as instantaneous spike rate, and then multiplied by dt to convert it to something we could interpret as the number of spikes in the given interval.  Thus, multiplying case (2) output by dt served to convert it to something that's somewhat consistent with case (3).

In the usual nonspiking LCA situation with a feedback loop between an error layer and a HyPerLCA layer, both the error layer and the LCA layer as well as the connection are working in case (2).  dt\_factor should be 1 in this case, no matter what dt is.

I propose that we do the following:
(a) If the connection uses stochastic release ( pvpatchAccumulateType = "stochastic"), then activity is scaled by dt before calculating the contribution to GSyn.  If necessary we can add a parameter that controls the factor to scale by.  This handles case (1) above.

(b) Get rid the of preActivityIsNotRate parameter.  Setting it to false in params generates a warning, and setting it to true generates an error.  Cases (2) and (3) are both handled correctly by dt\_factor=1, as long as we don't mix them in the same connection.

(c) If we anticipate having to convert case-(2) input to case (3) again (or vice versa), we could do one of two things.  Either the connection has a flag called something like convertRateToSpikeCount, or the layers could have flags called something like activityIsSpiking and expectsSpikingInput, and the connection could compare pre and post and convert accordingly.  convertRateToSpikeCount would, I guess, behave like preActivityIsNotRate does now, but I think it's a better name (and even better names than convertRateToSpikeCount no doubt exist).  This part would apply to pvpatchAccumulateType = "convolve" and, I suspect, to "maxpooling", but not to "stochastic", which is in part (a).

What are people's thoughts about those ideas?


Garrett Kenyon 
12/31/14

sounds right Pete.  I agree that there are relevant cases I didn't consider, especially the critical fact that different parts of the RHS of the update equation might need to be scaled differently (i.e. V might have to be scaled differently than GSyn).  So, I think you last proposed solution is the best one.  convertRateToSpikeCount is a very reasonable name for a flag but perhaps that flag should only be read if the pre and post layer indicate a possible ambiguity.  I don't want to add any more required params that are totally orthogonal to what the connection is doing 99% of the time.  Thus, HyPerLayer has a flag activityIsSpiking that defaults to false and HyPerConn only reads convertRateToSpikeCount if the pre activityIsSpiking flag is true.    

So, what would your proposed solution mean for LCA?

we would presumably do nothing.  all flags default to false.  we should get reasonable answers as we change dt.  As a test, dt = 1, 0.1, 0.01, 0.001 should converge to  some asymptotic value.

Now, if we were making a new subclass of HyPerLayer for modeling an LCA layer that received spiking input, we'd write a new update\_state method that treated V and GSyn differently on the RHS of the update\_state eq (i.e. we'd scale V in the usual way but we wouldn't scale GSyn).

And if we wanted to drive our new subclass with a non-spiking input, we'd set the convertRateToSpikeCount flag to true and the PetaVision Gods would take care of the rest...


Pete Schultz 
12/31/14

Should that be, "HyPerConn only reads convertRateToSpikeCount if the
pre activityIsSpiking flag is false"?  If activityIsSpiking is true, it's already a spike count so it wouldn't need to be converted.

If I have that right, HyPerConn would check two things before it adjusts GSyn.  If the pre's activityIsSpikingFlag is true, there's no rescaling even if convertRateToSpikeCount is set.  If convertRateToSpikeCount is false, there's no rescaling even if the pre layer is spiking.  If activityIsSpikingFlag is false but convertRateToSpikeCount is true, then we need to multiply by dt.

It occurs to me that the params file doesn't have to set the activityIsSpiking flag; the type of layer determines whether the flag should be true or not.  So LIF and SpikingLCA would set the flag to true, ANNLayer and Image would set it to false, and Retina would set it to the value of its spikingFlag parameter.  I can't think of any other layer subclasses that are sometimes spiking and sometimes not.

So we need to
(1) add a public virtual HyPerLayer method (but not a new HyPerLayer params file entry), say activityIsSpiking()
(2) replace the connection parameter preActivityIsNotRate with convertRateToSpikeCount
(3) make sure that we scale only when pre->activityIsSpiking() is false and convertRateToSpikeCount is true

Is that correct?


Garrett Kenyon
12/31/14

That's fine, with the Conn flag default being false. 

