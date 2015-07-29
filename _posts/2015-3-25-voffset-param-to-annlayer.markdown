---
layout: post
title:  "Add VOffset Parameter to ANNLayer"
date:   2015-3-25 23:01:55
author: Garrett Kenyon
categories: jekyll update
---

hey,

I want to add a param to ANNLayer, VOffset, which would simply add VOffset to whatever the initial value of V is.  With this additional parameter, it would be possible to use ANNLayer to implement a step function.

{% highlight text %}
VThresh =  1.0;
VOffset   =  1.0;
AMax      =  1.0;
AMin       =  0.0;
{% endhighlight %}


Pete Schultz
3/25/15

I believe that you could implement a step function currently by setting AMax and AShift.  For example, setting AMax to VThresh and AShift to zero would give A=AMin below V=VThresh and A=VThresh above V=Vthresh.  To have V jump up to a different value from VThresh, setting AShift shifts everything with V>VThresh+VWidth by that amount.

A while ago we talked about being able to specify a piecewise linear transfer function by specifying the endpoints of the line segments involved, but then it got lost in the shuffle.  I think it would be more intuitive to create Gar's example below by having the params specify the endpoints V=-infinity, A=0; V=1, A=0; V=1, A=1; and V=infinity, A=1.  We could use array parameters for that.  With the current scheme, each parameter individually has a simple enough description, but the overall result depends on the order in which the parameters are applied, and I find that difficult to remember: do we apply AMax and then AShift or the other way around?


Garrett Kenyon
3/25/15

ok, so to get a step function with a threshold of 0 we set ...

{% highlight text %}
VWidth      =  0.0;
VThresh    =   0.0;
AMax        =   1.0; 
AMin         =   0.0;
AShift       =    -1.0;
{% endhighlight %}

I believe that we subtract AShift from A so I'm guessing we need a negative AShift to get an increase.   Anyway, you're right, very hard to think about...





