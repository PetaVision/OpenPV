---
layout: post
title:  "PetaVision parameter generation"
date:   2014-03-14 23:01:55
author: Pete Schultz
categories: jekyll update
---

Hello all,

A while ago some of us discussed having the code for PetaVision do the following:

a) A PetaVision run should generate a params file that shows the params used, including params that were set to their default values.
b) The code should have documentation for each parameter, that can be generated automatically, either as comments to the generated params file or as part of the PetaVision documentation.

I have finished a refactoring that does (a), and prepares the ground for (b).  The changes to the  in the branches/ioparams directory on the repository.  I'd like to reintegrate this branch into the trunk at the end of the day on Monday.  If there are additional commits to the trunk before 5:00 on Monday, I will merge them into ioparams before committing to the trunk, but if changes are committed after five but before merging in the new commit, they may get clobbered.

I tried to keep the meaning and behavior of the parameters the same.  All the systems tests pass with necessary changes; and if the generated params files are then used as input, the tests still pass.   However, there are over 600 places in the current repository version where parameters are read, many with unique dependencies and peculiarities (several params were added just during the lifetime of this branch).  So if I got 99% of them right --- likely very optimistic --- I still broke half a dozen things.  Please let me know if you run into problems.

Here is an outline of the impending changes:

HyPerCol has a new string parameter, "printParamsFilename", that indicates the file that the params are written to.  The default is "pv.params"; paths that don't begin with a slash are taken to be relative to the outputPath directory.  The generated file has the format of a params file and can be used directly as input of a petavision run.  This replaces the <outputPath>/params.pv file which was only a dump of the PVParams object.

Layers, connections and probes have public constructors and initialize methods that take only the following arguments: the name of the object and the HyPerCol that it will be attached to.  Everything else that was passed as an argument to the constructor is now read from params.  Hence buildandrun.cpp is significantly simpler.

!!! If you have custom objects that subclass from one of these objects whose constructor and initialize() changed interfaces, they will need to be fixed !!!
I tried to make it so that this would be the only change that you would need to make if you're using the buildandrun functions to generate the column from the params file.

Objects that read from the params file have a virtual method ioParamsFillGroup(), that takes one argument, a switch that indicates whether reading or writing a params file.  Those functions call a sequence of functions with the name ioParam\_<parameterName>.  For example, ioParam\_nxScale in HyPerLayer reads/writes the nxScale parameter.  (The underscore after ioParam is so that the parameter name can appear in the function name exactly as it is capitalized in the parameter file, while still keeping the start of the parameter name clear.)

The reason for having a single method handle both reading the parameter and writing it to the printParamsFilename is so that as params are added, the input, output, and documentation aspects of the param appear in a single place.  The reason for a standardized ioParam\_ name is to make it easy to find params when generating documentation.

These functions are virtual so that subclasses can change the behavior of these functions.  For example, there are several KernelConn parameters that TransposeConn does not need, so those ioParam\_ methods do not read/write a parameter.  If it only makes sense to read a parameter if other parameters have certain values, that decision takes place in the ioParam\_ method, not ioParamsFillGroup.  This is so that subclasses can have complete control over the decision of whether a param is needed.

As a general rule, the ioParam\_ functions are called with the read switch during the initialize phase, and with the write switch after the allocateDataStructures phase but before the advanceTime loop.  However, there are some situations where the current behavior requires knowledge of other objects' params.  In these cases, those parameters must wait until the communicateInitInfo stage to be read.  I believe the only situations where that occurs are InitGauss2DWeights, where reading numOrientations{pre,post} and several related parameters depends on the number of features of the {pre,post} layer; and InitMTWeights, where several parameters are read only if nfp > 1, but nfp is often inferred from the postsynaptic nf.

Many HyPerConn functions that took filename as an argument now do not do so.  The InitWeightsFile parameter is handled by InitWeights.  All InitWeights methods, not just FileWeights, will use the param initWeightsFile.  The idea here, at Gar's suggestion, is that any weight initialization can read some of the weights from a file and calculate others using the weight initialization method.  This feature hasn't been implemented yet, though.

Again, I plan to commit this branch to the trunk Monday evening.  Please let me know of any concerns or problems.

Thanks.

Pete

