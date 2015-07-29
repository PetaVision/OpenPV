---
layout: post
title:  "dWMax is Required if plasticityFlag is True"
date:   2015-4-14 23:01:55
author: Pete Schultz
categories: jekyll update
---

Hello everyone.  I just made a commit that changes dWMax to a required parameter if plasticityFlag is true.  (If plasticityFlag is false and dWMax is defined in the params file, it will not be read and a warning will be issued that the parameter was skipped.)

Also, initialize\_base now initializes dWMax to NaN.  This is to catch any circumstances where the dWMax member variable is used but the dWMax parameter is not read — this should not happen.

