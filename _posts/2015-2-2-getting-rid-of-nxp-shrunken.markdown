---
layout: post
title:  "Getting Rid of nxp and nyp Shrunken"
date:   2015-2-2 23:01:55
author: Pete Schultz
categories: jekyll update
---

Hello everyone.  Up until now, patch sizes have had the following restrictions on nxp and nyp:
odd for a one-to-one connection;
an odd multiple of "many" for a one-to-many connection;
odd for a many-to-one connection.

We can, however, use nxpShrunken and nypShrunken to use any multiple of "many", even or odd, for a one-to-many connection.  The drawbacks to doing so are: (1) the patches still take the unshrunken size in memory and in disk storage; (2) we are probably missing a couple of places where we crawl over the whole patch when we could crawl over only the shrunken part; and (3) we are now using the phrase "shrunken patch" for three different concepts (adjusting patches at the edge of the local column, applying the shrinkPatches\_flag, and using nxpShrunken/nypShrunken).

I am about to commit a change that eliminates nxpShrunken and nypShrunken altogether.  Instead, nxp and nyp would be allowed to take any multiple, even or odd, for one-to-many connections.  The patch size would not be padded in memory or on disk.  For a one-to-four connection, for example, nxp could be 8 (in the current version nxp would have to be 12 and nxpShrunken would be 8).  The transpose of such a connection would have nxp=2.  So nxp for many-to-one connections is no longer required to be odd.  The attached file illustrates the connectivity in these situations.

If you set nxp and nxpShrunken in the params file, then nxp will be set to your value of nxpShrunken, and the specified value of nxp will not be used.  In the output params file specified by printParamsFilename, there will be an entry for nxp, but not for nxpShrunken.

The change passes all the system tests, and when I look at the internals in a debugger, it looks correct. One potential issue is if you generated weights with a version before this commit and used nxpShrunken.  Say, for example, you had nxp=14 and nxpShrunken=12.  If you then update and try to use that weight file in a new run or continue from a checkpoint, then the new run will have nxp=12, so there will be an error since the patch size of the weights file and the connection in memory don't agree.  If you change the params file by deleting nxpShrunken (or setting it to 14), the file can be read, but there is nothing in the code that will prevent the previously ignored region of the patch from being used or updated.  (The unused patch weights are probably zero so if the run is not updating weights, you're probably okay.)

To get around that issue, I added an m-file to mlab/util, called resizePatches.m

For a shared-weights file, the octave command would be
resizePatches(pathToOldPvpFile, pathToNewPvpFile, new\_nxp, new\_nyp);
For the example above, new\_nxp would be 12.  There are optional arguments to allow you to choose which part of the old patch to select; but for resizing because of the nxpShrunken/nypShrunken change, the defaults are correct.

For a nonshared-weights file, the command would be
resizePatches(pathToOldPvpFile, pathToNewPvpFile, new\_nxp, new\_nyp, nxGlobalPost, nyGlobalPost);
nxGlobalPost and nyGlobalPost are the global dimensions of the restricted postsynaptic layer.  They're needed for nonshared weight files because they aren't stored in the header but are necessary arguments to writepvpweightfile.m, which resizePatches.m uses.

