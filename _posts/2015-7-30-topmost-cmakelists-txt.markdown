---
layout: post
title:  "My Awesome Blog Post"
date:   2015-7-29 20:21:10
author: Pete Schultz
categories: jekyll update
---
Since the `CMakeLists.txt` in the topmost directory is now on the repository, there needs to be a way to keep this file from being committed and recommitted every time someone changes the list of projects they're working on.

Accordingly, there is now an entry in .gitignore for a file called `subdirectories.txt`, also in the topmost directory.  Instead of hardcoding the list of add_subdirectory commands, `CMakeLists.txt` now loops through that file and adds the directories it finds.

The file `sample-subdirectories.txt` illustrates the format.  Right now the format is pretty barebones - one subdirectory on a line, and each line contains a subdirectory and nothing else.  However, it might be worthwhile to allow for comments and whitespace.

Pete
