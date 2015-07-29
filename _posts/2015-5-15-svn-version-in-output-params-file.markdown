---
layout: post
title:  "SVN Version in the Output Params File"
date:   2015-5-15 23:01:55
author: Pete Schultz
categories: jekyll update
---

Hello everyone.  As you may have noticed, when PetaVision produces an output params file, the first line of the output pv.params file has been something like

{% highlight text %}
// PetaVision version something-point-something run at Fri May 15 17:10:33 2015
{% endhighlight %}

A while ago there was discussion about having the code try to automatically determine the svn version, and if it could be found, print that instead of “something-point-something”.  Yesterday I commited a change so that it will instead print something like
{% highlight text %}
// PetaVision, svn repository version 10090, run at Fri May 15 17:12:20 2015
{% endhighlight %}

If there are uncommitted modifications in trunk, the message is instead
{% highlight text %}
// PetaVision, svn repository version 10090 with local modifications, run at (timestamp)
{% endhighlight %}

The way I did this is to have *trunk/CMakeLists.txt* use the *cmake execute_process* command to call svnversion, and set the variable *PV_SVN_VERSION* based on the result.  That variable is then copied as a preprocessor directive into *src/include/cMakeHeader.h*.

The drawback of this approach is that just running make isn’t enough to update the version number; you have to run cmake as well as make.  It may be that using *add_custom_command/add_custom_target* is a better approach.  I’d prefer not to have PetaVision itself call svnversion at runtime, but that would be another possibility.

