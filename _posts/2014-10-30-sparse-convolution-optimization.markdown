---
layout: post
title:  "Sparse Convolution Optimization"
date:   2014-10-30 23:01:55
author: Sheng Lundquist
categories: jekyll update
---

I've just committed a big update on how PV deals with sparse layers. Recv from pre on the cpu now loops over active indices as opposed to everything. Prelim testing says I've gotten a speedup of 2 on the cpu. Let me know your speedup.

Sheng

