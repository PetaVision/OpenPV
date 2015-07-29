---
layout: post
title:  "New Interface to buildandrun"
date:   2015-1-5 23:01:55
author: Pete Schultz
categories: jekyll update
---

Hello everyone.  We've discussed having PetaVision organized into a core functionality and then additional libraries, instead of having the trunk become ever more and more complicated.  To accomodate this, I've added a new interface for buildandrun, as well as the main functions it calls, buildandrun1paramset and build.  When you get this update to the trunk, also be sure to update KernelTest and FourByFourGenerativeTest.  You will also need to run cmake since there are new files being added to trunk and to KernelTest.

First of, the current interface still exists so things should be backwards compatible.

However, the three functions named above have been overloaded, where the function pointer to customgroups is replaced with a pair of arguments.  The first is of type ParamGroupHandler **, which is an array of pointers to ParamGroupHandler objects; and the second is an integer that specifies the length of the array.

ParamGroupHandler is an abstract class in the trunk/io directory.  Only subclasses can be instantiated, and the subclass must implement a function called createObject.  It returnns a pointer to void and takes three arguments: a string indicating a keyword, a string indicating a group name, and a pointer to a HyPerCol.  The createObject decides whether it recognizes the given keyword; and if so, it creates an object in the given HyPerCol with the given name.  It then returns a pointer to the created object.  It returns a pointer to void because the object could be a layer, a connection, a probe, or something else.  If it doesn't recognize the keyword, it should return NULL, without a warning or error.  This is because there can in principle be several ParamGroupHandler objects, and it is an error only if none of them recognize the keyword.

CoreParamGroupHandler is a derived class of ParamGroupHandler that handles the groups in trunk.    It provides an example of implementing ParamGroupHandler.  Most of the functions in buildandrun.cpp has been marked obsolete, as they were only called by other functions in buildandrun, and those function calls have been replaced by calls to createObject methods.

I rewrote KernelTest to use a subclass of ParamGroupHandler instead of a customgroups function, which can also serve as an example of a ParamGroupHandler subclass.  In terms of the number of keywords to handle, the KernelTestGroupHandler method is probably more typical than the CoreParamGroupHandler one.

FourByFourGenerativeTest had an unnecessary function call to one of the buildandrun functions that is now obsolete.  That call has been removed, so that test needs to be updated or there will be linking errors.

There are other system tests that use a customgroups function pointer to buildandrun.  I left them alone for now, as the changes are backwards compatible.

Please let me know if you run into any problems with this update.

