---
layout: post
title:  "Standard Names for Connections"
date:   2015-3-16 23:01:55
author: Pete Schultz
categories: jekyll update
---

Hello everyone.  For a while, connections have had the property that if their name had the form "PreLayerName to PostLayerName” the preLayerName and postLayerName didn’t have to be specified in the params file.  (if preLayerName and postLayerName are specified, they overrule the specification, so that a HyPerConn called “One to One Connection” doesn’t have to have preLayerName “One" and postLayername "One Connection").  The trouble with this is that checkpointing creates filenames based on the name of the layer or connection, and spaces in filenames can cause problems.

Many people have been using “PreLayerNameToPostLayerName” as the standard name for connections, and that’s what Max’s params extractor scans for.  So I think it would make sense for the inferPreAndPostFromConnName() method to also use "To" instead of " to “.  Would it cause trouble for anyone if I made this change?

