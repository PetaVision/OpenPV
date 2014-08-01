/*
 * LIFGap.cpp
 *
 *  Created on: Jul 29, 2011
 *      Author: garkenyon
 */

#include "LIFGap.hpp"
#include "../include/pv_common.h"
#include "../include/default_params.h"
#include "../io/fileio.hpp"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#ifdef __cplusplus
extern "C" {
#endif

void LIFGap_update_state_original(
    const int numNeurons,
    const float time,
    const float dt,

    const int nx,
    const int ny,
    const int nf,
    const int nb,

    LIF_params * params,
    uint4 * rnd,

    float * V,
    float * Vth,
    float * G_E,
    float * G_I,
    float * G_IB,
    float * GSynHead,
    float * activity,

    const float sum_gap
);

void LIFGap_update_state_beginning(
    const int numNeurons,
    const float time,
    const float dt,

    const int nx,
    const int ny,
    const int nf,
    const int nb,

    LIF_params * params,
    uint4 * rnd,

    float * V,
    float * Vth,
    float * G_E,
    float * G_I,
    float * G_IB,
    float * GSynHead,
    float * activity,

    const float sum_gap
);

void LIFGap_update_state_arma(
    const int numNeurons,
    const float time,
    const float dt,

    const int nx,
    const int ny,
    const int nf,
    const int nb,

    LIF_params * params,
    uint4 * rnd,

    float * V,
    float * Vth,
    float * G_E,
    float * G_I,
    float * G_IB,
    float * GSynHead,
    float * activity,

    const float sum_gap
);


#ifdef __cplusplus
}
#endif


namespace PV {

LIFGap::LIFGap() {
   initialize_base();
}

LIFGap::LIFGap(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc, TypeLIFGap, "LIFGap_update_state");
#ifdef PV_USE_OPENCL
   if(gpuAccelerateFlag)
      initializeGPU();
#endif
}

LIFGap::LIFGap(const char * name, HyPerCol * hc, PVLayerType type) {
   initialize_base();
   initialize(name, hc, type, "LIFGap_update_state");
#ifdef PV_USE_OPENCL
   if(gpuAccelerateFlag)
      initializeGPU();
#endif
}

LIFGap::~LIFGap()
{


#ifdef PV_USE_OPENCL
   if(gpuAccelerateFlag) {
      delete clG_Gap;
      delete clGSynGap;
   }
#endif

}

int LIFGap::initialize_base() {
   numChannels = 4;
#ifdef PV_USE_OPENCL
   clG_Gap = NULL;
   clGSynGap = NULL;
#endif

   return PV_SUCCESS;
}

// Initialize this class
/*
 *
 */
int LIFGap::initialize(const char * name, HyPerCol * hc, PVLayerType type, const char * kernel_name) {
   int status = LIF::initialize(name, hc, type, kernel_name);

#ifdef PV_USE_OPENCL
   numEvents=NUM_LIFGAP_EVENTS;
#endif

   return status;
}


#ifdef PV_USE_OPENCL
/**
 * Initialize OpenCL buffers.  This must be called after PVLayer data have
 * been allocated.
 */
int LIFGap::initializeThreadBuffers(const char * kernel_name)
{
   int status = CL_SUCCESS;

   status = LIF::initializeThreadBuffers(kernel_name);

   const size_t size    = getNumNeurons()  * sizeof(pvdata_t);
   CLDevice * device = parent->getCLDevice();

   // these buffers are shared between host and device
   //

   clGSynGap = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, getChannel(CHANNEL_GAP));

   return status;
}

int LIFGap::initializeThreadKernels(const char * kernel_name)

{
   int status = CL_SUCCESS;

   status = LIF::initializeThreadKernels(kernel_name);

   int argid = getNumKernelArgs();

   status |= krUpdate->setKernelArg(argid++, sumGap);
   status |= krUpdate->setKernelArg(argid++, clG_Gap);
   //status |= krUpdate->setKernelArg(argid++, clGSynGap);

   numKernelArgs = argid;

   return status;
}
#endif

int LIFGap::allocateConductances(int num_channels) {
   this->sumGap = 0.0f;
   return LIF::allocateConductances(num_channels-1); // Don't need conductance for gap channel
}

int LIFGap::updateStateOpenCL(double time, double dt)
{
   int status = CL_SUCCESS;

#ifdef PV_USE_OPENCL
   status = LIF::updateStateOpenCL(time, dt);

#if PV_CL_COPY_BUFFERS
   status |= clGSynGap->copyFromDevice(1, &evUpdate, &evList[getEVGSynGap()]);
#endif

   //do we need to copy gap back and forth?
   //status |= clG_Gap->copyFromDevice(1, &evUpdate, &evList[EV_LIF_GSYN_GAP]);
//   status |= getChannelCLBuffer(CHANNEL_GAP)->copyFromDevice(1, &evUpdate, &evList[getEVGSynGap()]);
//   numWait += 1;
#endif

   return status;
}

int LIFGap::updateState(double time, double dt)
{
   int status = CL_SUCCESS;
   update_timer->start();

#ifdef PV_USE_OPENCL
   if((gpuAccelerateFlag)&&(true)) {
      updateStateOpenCL(time, dt);
   }
   else {
#endif

   const int nx = clayer->loc.nx;
   const int ny = clayer->loc.ny;
   const int nf = clayer->loc.nf;
   const int nb = clayer->loc.nb;

   pvdata_t * GSynHead   = GSyn[0];
   pvdata_t * activity = clayer->activity->data;

   switch (method) {
   case 'a':
      LIFGap_update_state_arma(getNumNeurons(), time, dt, nx, ny, nf, nb, &lParams, randState->getRNG(0), clayer->V, Vth, G_E,
            G_I, G_IB, GSynHead, activity, sumGap);
   break;
   case 'b':
      LIFGap_update_state_beginning(getNumNeurons(), time, dt, nx, ny, nf, nb, &lParams, randState->getRNG(0), clayer->V, Vth, G_E,
            G_I, G_IB, GSynHead, activity, sumGap);
   break;
   case 'o':
      LIFGap_update_state_original(getNumNeurons(), time, dt, nx, ny, nf, nb, &lParams, randState->getRNG(0), clayer->V, Vth, G_E,
            G_I, G_IB, GSynHead, activity, sumGap);
      break;
   default:
      assert(0);
      break;
   }
#ifdef PV_USE_OPENCL
   }
#endif
   updateActiveIndices();
   update_timer->stop();
   return status;
}


} // namespace PV

///////////////////////////////////////////////////////
//
// implementation of LIF kernels
//

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PV_USE_OPENCL
#  include "../kernels/LIFGap_update_state.cl"
#else
#  undef PV_USE_OPENCL
#  include "../kernels/LIFGap_update_state.cl"
#  define PV_USE_OPENCL
#endif

#ifdef __cplusplus
}
#endif
