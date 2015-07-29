---
layout: post
title:  "Changes to CMakeLists files"
date:   2015-4-7 23:01:55
author: Pete Schultz
categories: jekyll update
---

Hello everyone.  I just added a commit that makes it an error to omit the *-t* command line option if you’ve compiled with *PV_USE_OPENMP_THREADS* defined.

The behavior with the -t option is now as follows:
If you’ve compiled without threads, you can specify *-t 1*, but any other use of *-t* will give an error (this is the same as previously).
If you’ve compiled with threads, you must have *-t* in the command line.
*-t <positive number>* specifies the number of threads explicitly.
*-t 0* or *-t* tells PetaVision to get the maximum number of threads from OpenMP and to use that number.

There must be whitespace between *-t* and its argument if there is one: *-t8*, for example, will give an error.

