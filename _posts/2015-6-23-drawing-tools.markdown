---
layout: post
title:  "Drawing Tools"
date:   2015-6-23 23:01:55
author: Max Theiler
categories: jekyll update
---

Just committed an overhaul of the python side of the 'draw' command. 'Infodraw' has been taken out for the moment, but 'draw' should be a lot more robust, legible, and extensible. It now understands the # and @ syntax for example, ignores comments and prints warnings about mislabeled connections and such. It also now color-codes layers by scale, rather than phase.

Usage remains the same, once you've followed the install instructions (attached, with legend) you should just be able to type "draw myfile.params" from the command line to get a .png of your params file. 

Hopefully draw will crash less often and be more useful as a debugging tool now that the user doesn't need to use a 'clean' output pv.params.


Max Theiler
6/26/15

Pete and I ran into a bit of trouble with the latest (0.5.1) version of mermaid throwing some funny warnings and not rendering graphs correctly. Specifically, it doesn't seem to draw arrowheads.

The workaround for the moment is to use the old version (0.4.0), which most of you probably have installed anyway. Meanwhile I'm looking into resolving the problems with the latest version.

You can check which version you have with

{% highlight bash %}
mermaid --version
{% endhighlight %}

If its not 0.4.0, I recommend you uninstall the new version with

{% highlight bash %}
sudo npm uninstall -g mermaid
{% endhighlight %}

then re-install the older version with:

{% highlight bash %}
sudo npm install -g mermaid@0.4.0
{% endhighlight %}

