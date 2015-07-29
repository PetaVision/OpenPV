---
layout: post
title:  "Checkpoint Supress with Plasticity Off"
date:   2015-1-28 23:01:55
author: Pete Schultz
categories: jekyll update
---

Hello.  On Monday night I added a flag to HyPerCol, to suppress writing checkpoints of connections with plasticityFlag off.  However, the change I made to HyPerCol::checkpointWrite to implement that flag was wrong, and it suppressed checkpointing all weights, plastic or not, if the flag was set to false, which is the default.  If you updated the trunk on Tuesday or this morning, you'll need to update again now, or checkpoints won't contain any of the weights.  It should be fixed now, but please let me know if you have any other issues with this.

I also added a line to the CheckpointSystemTest to delete the old checkpoint and output directories before doing anything else.  That would have caught the bug, since CheckpointSystemTest was comparing the files written during an earlier run.  Please make sure to update that system test as well.

