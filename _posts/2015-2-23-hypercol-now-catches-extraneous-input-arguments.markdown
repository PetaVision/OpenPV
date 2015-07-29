---
layout: post
title:  "HyPerCol Now Catches Extraneous Input Arguments"
date:   2015-2-23 23:01:55
author: Pete Schultz
categories: jekyll update
---

Hello everyone.  Up to now, if extra command line arguments were passed to HyPerCol, they would be ignored.  This can cause problems if an option is mistyped, leading to PetaVision running but not in the intended way.  The most recent update causes HyPerCol to exit with a failure return code in this case, and the error message prints out which argument(s) were not understood.

If your project merely calls buildandrun, there should not be any changes necessary.  However, the parse\_options function and the various pv\_getopt\* functions in io.c have additional arguments.  Several system tests use these functions, so they will need to be updated along with trunk.

Here are the details:
parse\_options and the pv\_getopt\* functions now take a pointer to an array of booleans, of length argc, called paramusage.  parse\_options initializes the array to all false. It passes the array to each pv\_getopt\* function, which marks parameters as read as they are parsed.  After HyPerCol calls parse\_options, it checks whether any of the input parameters were unused.  parse\_options sets the array of booleans but does not check it.  The pv\_getopt\* functions will ignore paramusage if it is null, but the parse\_options function requires it (should that be changed?)

