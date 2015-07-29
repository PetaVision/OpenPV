---
layout: post
title:  "BaseConnection and BaseConnectionProbe"
date:   2014-10-29 23:01:55
author: Pete Schultz
categories: jekyll update
---

Hello everyone.  I don't have byte weights finished yet, but I'll be making some commits that prepare the ground for adding that capability.  I just made the first such commit.

BaseConnection is a new superclass for the connection hierarchy, with HyPerConn the only class directly derived from it.  HyPerCol now maintains a list of BaseConnections instead of a list of HyPerConns.  This means that when HyPerConn becomes a template, HyPerCol does not need to worry which connections have been instantiated as floating-point and which have been instantiated as bytes.

HyPerLayer should also use only references to BaseConnections, not HyPerConns, for the same reason.  However, currently the recvSynapticInput method and related methods take HyPerConns as arguments and refer to the connections' accumulate-function pointers.  I am planning to move those methods to the connection hierarchy, so that HyPerLayer doesn't need to call function templates, and therefore won't need to be aware of how the connection is implemented.  In my scratch workspace I have done this for the CPU receive methods, but not the GPU methods.

Before this latest commit, BaseConnectionProbe had several subclasses.  There is now a BaseHyPerConnProbe between BaseConnectionProbe and those classes.  The plan, similar to BaseConnection/HyPerConn, is to have BaseConnectionProbe be untemplated, so that you can refer to a connection probe without specifying a weight type, and that BaseHyPerConnProbe will be templated.  BaseConnectionProbe has a method getTargetConn that returns a BaseConnection, while BaseHyPerConnProbe has a method getTargetHyPerConn that returns the same pointer, but typed as a HyPerConn.

The new commit passes all the systems tests.  Since there are new .cpp files added, you'll have to run cmake or cmake28 before compiling with the new version.


