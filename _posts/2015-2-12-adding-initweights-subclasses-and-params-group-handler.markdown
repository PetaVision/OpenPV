---
layout: post
title:  "Adding InitWeights Subclasses and the ParamsGroupHandler Interface"
date:   2015-2-12 23:01:55
author: Pete Schultz
categories: jekyll update
---

Hello everyone.  Up to now, the InitWeightType parameter was decoded by HyPerConn, with the allowed InitWeightType strings hard-coded in HyPerConn::createInitWeightsObject.  This meant that adding a new InitWeights method, no matter how specialized, required modifying HyPerConn in trunk.  This needed to change, so that a sandbox could have specialized connections and/or initweights methods.  We want to be able to use an init weights method with a connection regardless of whether the initialization method comes from trunk or from the sandbox, and regardless of whether the connection comes from trunk or sandbox.

I've committed changes to make this possible.  The commit affects the trunk as well as several system tests, and all the system tests pass. I've tried to keep things as backwards-compatible as possible, but there are some changes that are necessary.

* The constructors for HyPerConn and for most of its subclasses take 4 arguments instead of 2.  The third argument for HyPerConn's constructor is a pointer to an InitWeights object, and the fourth argument is a pointer to a NormalizeBase object.  During initialization, their values are assigned to the member variables weightInitializer and normalizer.  If either is constructor argument is null, the HyPerConn determines the corresponding member variable, with results that should be the same as before.  Some subclasses don't use the initWeightType parameter (e.g. clones, transposes, IdentConn), and their constructors don't take a weightInitializer constructor argument.  Similarly for the normalizeMethod parameter and the weightNormalizer constructor argument.

If you have any subclasses of HyPerConn in your sandbox, you should add weightInitializer and weightNormalizerarguments to the constructor, and have the initialize method pass those values to the parent class's initialize method.

- The ParamGroupHandler interface changed significantly.  If you have a ParamGroupHandler subclass, you'll need to change it accordingly.  KernelTest has an example in src/KernelTestGroupHandler.{c,h}pp, and StochasticReleaseTest has an example as part of src/main.cpp.  The changes are:

- A subclass of ParamGroupHandler should have a getGroupType method that takes a string as input and returns whether the string is a keyword for a layer, a connection, etc.  There is a return type for a keyword that is unrecognized.

- Also, instead of ParamGroupHandler::createObject, there are separate methods createLayer, createConnection, createProbe, createWeightInitializer, and createWeightNormalizer.  Their prototypes are in io/ParamGroupHandler.hpp.  If these methods get a keyword that doesn't correspond to a known keyword of the appropriate type, the method should return NULL without an error.  If it recognizes the keyword it should create the object.  If creating the object fails, that is an error.

It should not be necessary to change the call to buildandrun.  Internally, it creates connections differently.  This is to allow for the fact that the relevant connection class and InitWeights class may come from different sources (i.e. trunk and sandbox, or two different libraries).  buildandrun() reads the weightInitType parameter, and tries to find a ParamGroupHandler that knows that type.  It then calls that handler's createWeightInitializer to create an InitWeights object.  Finally, it passes the resulting InitWeights object to createConnection.  The same thing will soon happen with normalizeMethod, but for now CoreParamGroupHandler::createWeightNormalizer always returns NULL.

