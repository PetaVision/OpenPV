---
layout: post
title:  "Cleaning old classes out of trunk"
date:   2014-12-22 23:01:55
author: Pete Schultz
categories: jekyll update
---

Hello.  As an experiment I removed everything from src/trunk and then put back only enough to compile trunk and all the system tests.

With CUDA acceleration on
With OpenCL acceleration on
Without either gpu acceleration method.

I also scanned the active sandbox params files (but did not try to build the sandboxes) and included any classes mentioned in the params files.

I added the minimal number of files necessary to build and run the system tests either with or without CUDA (tested on NMC machines), and either with or without OpenCL (tested on my local mac).

The following files listed below were NOT necessary.

I think I'll keep arch/mpi/mpi.\*, since its purpose is to provide stubs for MPI functions when MPI is not present.  Right now we require MPI and compiling probably breaks if MPI is turned off, but it might be good to have MPI be optional.  Xinhua just added KmeansLayer, so that should stay even though it isn't being used in any sandbox on the repository yet.  Also, Gar has asked to keep the clique classes.  Is there anything else in the list below that needs to stay in trunk?

{% highlight text %}
arch/cuda/device_util.hpp
arch/mpi
arch/mpi/mpi.c
arch/mpi/mpi.h
arch/opencl/cl_info.cpp
arch/openclMain
arch/openclMain/convolve.cl
arch/openclMain/convolve_cpu.cl
arch/openclMain/convolve_main.cpp
arch/pthreads
arch/pthreads/pv_thread.c
arch/pthreads/pv_thread.h
arch/rr
arch/rr/HyPerLayer.f90
arch/rr/HyperLayer.c
arch/rr/LIF.f90
arch/rr/layers
arch/rr/layers/elementals.f90
arch/rr/layers/elementals.h
arch/rr/ppu
arch/rr/ppu/pv_ppu.c
arch/rr/pv_cell.h
arch/rr/pv_ppu.c
arch/rr/pv_spu.c
arch/rr/spu
arch/rr/spu/LIF_spu.c
arch/rr/spu/main_spu.c
arch/rr/spu/pv_spu.c
arch/rr/vectorizables.f90
build
build/ppu
build/pthreads
build/spu
connections/CliqueConn.cpp
connections/CliqueConn.hpp
connections/InhibSTDPConn.cpp
connections/InhibSTDPConn.hpp
connections/LCALIFLateralKernelConn.cpp
connections/LCALIFLateralKernelConn.hpp
connections/MapReduceKernelConn.cpp
connections/MapReduceKernelConn.hpp
connections/OjaKernelConn.cpp
connections/OjaKernelConn.hpp
connections/STDP3Conn.cpp
connections/STDP3Conn.hpp
connections/STDPConn.cpp
connections/STDPConn.hpp
connections/WindowConn.cpp
connections/WindowConn.hpp
include/cell.mk
include/clmake.mk
include/depend.mk
include/gl_stubs.h
include/mpi_stubs.h
include/neural_tuning.h
include/pv.h
include/sources.mk
input
input/amoeba2X.bin
input/amoeba2X.tif
input/circ_pix_input.bin
input/circle1_figure_0.bin
input/circle1_input.bin
input/circle1_num.bin
input/egg_input.bin
input/image_buildandrun_example.png
input/inparams_GTK.txt
input/params.cocirc
input/params.cocirc.ilya
input/params.gabor_cocirc
input/params.synch
input/params_buildandrun_example.pv
input/t64_input.bin
input/t64_red.tif
input/test128.bin
input/test256.bin
input/test64.bin
input/vertical-line.tif
input/w_center_surround_4x4.bin
input/w_center_surround_8x8.bin
io/ActivityProbe.cpp
io/ActivityProbe.hpp
io/ConnStatsProbe.cpp
io/ConnStatsProbe.hpp
io/GLDisplay.cpp.mpi
io/LCALIFLateralProbe.cpp
io/LCALIFLateralProbe.hpp
io/LinearActivityProbe.cpp
io/LinearActivityProbe.hpp
io/LinearAverageProbe.cpp
io/LinearAverageProbe.hpp
io/OjaConnProbe.cpp
io/OjaConnProbe.hpp
io/OjaKernelSpikeRateProbe.cpp
io/OjaKernelSpikeRateProbe.hpp
io/PVPFile.cpp
io/PVPFile.hpp
io/PatchProbe.cpp
io/PatchProbe.hpp
io/PointLCALIFProbe.cpp
io/PointLCALIFProbe.hpp
io/PostConnProbe.cpp
io/PostConnProbe.hpp
io/SparsityTermFunction.cpp
io/SparsityTermFunction.hpp
io/SparsityTermProbe.cpp
io/SparsityTermProbe.hpp
kernels/ANNDivLayer_update_state.cl
kernels/ANNLabelLayer_update_state.cl
kernels/ANNWeightedErrorLayer_update_state.cl
kernels/BIDS_update_state.cl
kernels/Conn_update_state.f90
kernels/HyPerLayer_recv_pre.cl
kernels/LIF_update_state.f90
kernels/Retina_update_state.f90
layers/ANNDivInh.cpp
layers/ANNDivInh.hpp
layers/ANNLabelLayer.cpp
layers/ANNLabelLayer.hpp
layers/ANNTriggerUpdateOnNewImageLayer.cpp
layers/ANNTriggerUpdateOnNewImageLayer.hpp
layers/ANNWeightedErrorLayer.cpp
layers/ANNWeightedErrorLayer.hpp
layers/AccumulateLayer.cpp
layers/AccumulateLayer.hpp
layers/CliqueLayer.cpp
layers/CliqueLayer.hpp
layers/FilenameParsingGroundTruthLayer.cpp
layers/FilenameParsingGroundTruthLayer.hpp
layers/KmeansLayer.cpp
layers/KmeansLayer.hpp
layers/MembranePotentialLayer.cpp
layers/MembranePotentialLayer.hpp
layers/NaiveBayesLayer.cpp
layers/NaiveBayesLayer.hpp
main_buildandrun_example.cpp
normalizers/NormalizeScale.cpp
normalizers/NormalizeScale.hpp
visual
weightinit/Init3DGaussWeights.cpp
weightinit/Init3DGaussWeights.hpp
weightinit/Init3DGaussWeightsParams.cpp
weightinit/Init3DGaussWeightsParams.hpp
weightinit/InitByArborWeights.cpp
weightinit/InitByArborWeights.hpp
weightinit/InitDistributedWeights.cpp
weightinit/InitDistributedWeights.hpp
weightinit/InitDistributedWeightsParams.cpp
weightinit/InitDistributedWeightsParams.hpp
weightinit/InitGaussianRandomWeights.cpp
weightinit/InitGaussianRandomWeights.hpp
weightinit/InitGaussianRandomWeightsParams.cpp
weightinit/InitGaussianRandomWeightsParams.hpp
weightinit/InitMTWeights.cpp
weightinit/InitMTWeights.hpp
weightinit/InitMTWeightsParams.cpp
weightinit/InitMTWeightsParams.hpp
weightinit/InitPoolWeights.cpp
weightinit/InitPoolWeights.hpp
weightinit/InitPoolWeightsParams.cpp
weightinit/InitPoolWeightsParams.hpp
weightinit/InitRuleWeights.cpp
weightinit/InitRuleWeights.hpp
weightinit/InitRuleWeightsParams.cpp
weightinit/InitRuleWeightsParams.hpp
weightinit/InitSubUnitWeights.cpp
weightinit/InitSubUnitWeights.hpp
weightinit/InitSubUnitWeightsParams.cpp
weightinit/InitSubUnitWeightsParams.hpp
weightinit/InitWindowed3DGaussWeights.cpp
weightinit/InitWindowed3DGaussWeights.hpp
weightinit/InitWindowed3DGaussWeightsParams.cpp
weightinit/InitWindowed3DGaussWeightsParams.hpp
{% endhighlight %}

