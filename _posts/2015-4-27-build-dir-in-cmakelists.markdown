---
layout: post
title:  "BUILD_DIR in CMakeLists"
date:   2015-4-27 23:01:55
author: Pete Schultz
categories: jekyll update
---

Hello everyone.  I just added a cmake variable, *BUILD_DIR*, to the various system tests’ CMakeLists.txt files.  It allows you to specify the directory that the executable binaries will be written to (it can create the directory if needed, but won’t create a subdirectory if the intended parent directory doesn’t exist).

Before, the directory was taken from *CMAKE_BUILD_TYPE*; that is, if BasicSystemTest had *CMAKE_BUILD_TYPE=Debug*, the executable would be *BasicSystemTest/Debug/BasicSystemTest*, etc.

Now, if *BUILD_DIR* is empty, the directory is taken from *CMAKE_BUILD_TYPE* as before.  But if *BUILD_DIR* is not empty, it will use that as the directory instead.  You can specify an absolute path, or a relative path; relative paths are relative to *CMAKE_CURRENT_SOURCE_DIR*, which in the BasicSystemTest example would be BasicSystemTest.


