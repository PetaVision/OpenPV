---
layout: post
title:  "New Specify GPU Flag"
date:   2015-5-11 23:01:55
author: Sheng Lundquist
categories: jekyll update
---

Hi all,

I've just made a commit that allows you to automatically adjust which GPU each MPI process should use. To specify this, you use the same flag as before, with *-d -1*.

What's new about this feature?
Previously, specifying *-d -1* set each MPI process rank to the GPU index (MPI rank 0 uses GPU 0, MPI rank 1 uses GPU 1). This would have not ran if the number of MPI processes was greater than the number of GPUs on the machine. Furthermore, there was no options to run across machines with multiple GPUs. In contrast, specifying the flag now will automatically assign a GPU to each MPI process depending on the host name and number of available GPUs on that host. Furthermore, PetaVision will now warn you if you are under or overloading each machine based on number of MPI processes vs number of GPUs on that machine.

Eventually, I hope that we can get rid of the *-1* flag (syntactically, it's a terrible flag) and have PetaVision default to this behavior (as opposed to defaulting to GPU 0) once this feature has been tested and verified by other people.

