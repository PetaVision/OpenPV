---
layout: post
title:  "Checkpoint Based on Clock Time"
date:   2015-3-25 23:01:55
author: Pete Schultz
categories: jekyll update
---

Hi everyone.  I just added a new option for the string parameter checkpointWriteTriggerMode: “clock”, which writes a checkpoint based on the clock time, as opposed to the simulation time or step number.

If checkpointWriteTriggerMode = “clock”, then two additional params are read: checkpointWriteClockInterval, which is numerical, and checkpointWriteClockUnit, which is a string.

For example, if the HyPerCol params group has

{% highlight text %}
checkpointWriteTriggerMode = "clock";
checkpointWriteClockInterval = 30.0;
checkpointWriteClockUnit = “minutes”;
{% endhighlight %}

then a checkpoint gets written on the first timestep after thirty minutes have passed since the last checkpoint was written.

The units for checkpointWriteClockUnit are “seconds”, “minutes”, “hours”, and “days”.  The code has some leeway about the parameter string; you can look at HyPerCol::ioParam\_checkpointWriteClockUnit() if you want to see what is recognized.  Internally, the string gets converted to one of the four strings above, and the standardized string is what gets printed to the output pv.params file.

