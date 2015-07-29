---
layout: post
title:  "Generating an Immediate Checkpoint"
date:   2015-7-7 23:01:55
author: Pete Schultz
categories: jekyll update
---

Hi everyone.  With Kendall’s help, I've committed changes to HyPerCol so that sending a signal causes HyPerCol to write an immediate checkpoint.  The original suggestion, of using control-C to send a *SIGINT*, doesn’t work under MPI --- mpiexec doesn’t propagate *SIGINT* to the processes it launched.  mpiexec does, however, propagate *SIGUSR1*.  I don’t think there’s a way to cause a keyboard press in the terminal running a process to send it a *SIGUSR1* signal.  So you’d need to use the kill system call.

The easiest way is with the killall command.

Launch a long-running, multitimestep petavision process under MPI, with the checkpointWrite parameter set to true, e.g.
mpiexec -np 4 Debug/BasicSystemTest -p paramsFileWithLargeStopTimeAndCheckpointingTurnedOn.params

While it’s running, switch to another terminal, and type either

{% highlight bash %}
killall -SIGUSR1 mpiexec
{% endhighlight %}

or

{% highlight bash %}
killall -SIGUSR1 BasicSystemTest
{% endhighlight %}

Either should generate a checkpoint immediately (if one time steps takes a while, “immediately” means once it reaches the end of the timestep).

The killall command sends the signal to all processes with the given name, so if you have other mpiexec processes and/or other BasicSystemTest processes running, you’ll have to take care that you don’t send a signal somewhere you don’t want it to go.  In that case, the easiest way is to find the PID of the mpiexec process and use

{% highlight bash %}
kill -SIGUSR1 <PID>
{% endhighlight %}

If you’re running under MPI with more than one process, only the root process will watch for the *SIGUSR1* signal.  That’s to eliminate any possibility of problems caused if one process checks for the signal just before it arrives, while another checks just after it arrives.

One thing I’ve noticed, that I’d like to fix:
Say that the checkpoint interval is 1000, and you send a signal at t=8421.  The checkpoints directory will then have checkpoints at 7000, 8000, 8421, 9000, 10000, 11000, etc.  But if you restart from checkpoint 8421, then checkpoints get written at 9421, 10421, 11421, etc.  I think it would be better for it to still checkpoint at 9000, 10000, etc.

