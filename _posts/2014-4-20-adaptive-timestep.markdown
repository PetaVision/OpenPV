---
layout: post
title:  "Adaptive Timestep"
date:   2014-04-20 23:01:55
author: Garrett Kenyon
categories: jekyll update
---

Mainly for Pete but whoever is interested,

my current plan for implementing a new adaptive time step is as follows:

1. add a loop in HyPerCol:advanceTime that will query all HyPerLayer::getDeltaTime(){return parent->getDeltaTimeBase();}

   set HyPerCol::deltaTime = minimum of HyPerLayer::getDeltaTime()

2. add a new class ANNNormalizedErrorLayer that will override getDeltaTime() to return the \|error\| / \|input\|

3. change calls in HyPerCol that use deltaTime to calculate startTime, stopTime, etc, to use deltaTimeBase (not strictly necessary but better coding practice).

I believe the above will cause the deltaTime to equal deltaTimeBase when \|error\|/\|input\| = 1.  We would presumably use a large value for timeConstantTau = ~100 - 400 so that we take very small time steps when the error is large.  

when \|error\|/\|input\| < 1, deltaTime will increase in direct proportion, potentially growing pretty large as the minimum \|error\|/\|input\| declines.  Note, we rarely see a minimum value of this quantity of less than 5-10%, so the effective dt/tau should rarely fall below 1.  

According to Walt, the above procedure converges in ~20 time steps for a single layer.  That would be a huge improvement over the current behavior.


Garrett Kenyon
4/29/14

to use adaptive time steps, change ANNErrorLayer to ANNNormalizedErrorLayer

and add the following to your HyPerCol params

dtAdaptFlag = true;
dtScaleMax = 5.0;
dtScaleMin = 0.25;
dtChangeMax = 0.05;
dtChangeMin = 0.0;

I'll send another email explaining what those number mean as soon as I have a moment.


Garrett Kenyon
7/10/14

Pete and I inadvertently introduced a bad bug into adaptTimeScale.  I think I've got it fixed now but please update to the latest version on the repository before starting any new runs and you'd better kill any current runs that are using the n-1th version.

the correct code block in adaptTimeScale should look like this:

{% highlight c %}
// if error is increasing, retreat back to the MIN(timeScaleMin, minTimeScaleTmp)
if (changeTimeScaleTrue < changeTimeScaleMin){
   if (minTimeScaleTmp > 0.0)
      timeScale =  minTimeScaleTmp < timeScaleMin ? minTimeScaleTmp : timeScaleMin;
   else{
      timeScale = timeScaleMin;
   }
}
{% endhighlight %}

the incorrect code block inserted by Pete and I looks like this (causes timeScale to jump discontinuously from current value to minTimeScaleTmp, which could be as large as timeScaleMax! if error is decreasing--big whoops--instability every time!):

{% highlight c %}
if (changeTimeScaleTrue < changeTimeScaleMin){
   timeScale =  minTimeScaleTmp > 0 ? minTimeScaleTmp : timeScaleMin;
}
{% endhighlight %}

the old code block, which "works", but doesn't let the timeScale go below timeScaleMin--which we want to allow if the error is particularly large--looks like this:
{% highlight c %}
if (changeTimeScaleTrue < changeTimeScaleMin){
   timeScale =  timeScaleMin;
}
{% endhighlight %}


BTW: assuming that the latest version of adaptTimeScale absolutely guarantees stability (I think it does--or should), I'm hoping we can use a much larger dWMax.  I'm trying that now (dWMax = 1.0 -> dWMax = 10.0).


