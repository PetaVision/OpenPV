/*
 * LIF.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: C. Rasmussen
 */

#include "../include/pv_common.h"
#include "../include/default_params.h"
#include "../connections/PVConnection.h"
#include "../utils/cl_random.h"
#include "HyPerLayer.hpp"
#include "LIF.hpp"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void LIF_update_state(
    const float time,
    const float dt,

    const int nx,
    const int ny,
    const int nf,
    const int nb,

    const LIF_params * params,
    uint4 * rnd,

    float * V,
    float * Vth,
    float * G_E,
    float * G_I,
    float * G_IB,
    float * phiExc,
    float * phiInh,
    float * phiInhB,
    float * activity);

#ifdef __cplusplus
}
#endif

namespace PV
{

#ifdef OBSOLETE
LIFParams LIFDefaultParams =
{
    V_REST, V_EXC, V_INH, V_INHB,            // V (mV)
    TAU_VMEM, TAU_EXC, TAU_INH, TAU_INHB,
    VTH_REST,  TAU_VTH, DELTA_VTH,	     // tau (ms)
    250, 0*NOISE_AMP*( 1.0/TAU_EXC ) * ( ( TAU_INH * (V_REST-V_INH) + TAU_INHB * (V_REST-V_INHB) ) / (V_EXC-V_REST) ),
    250, 0*NOISE_AMP*1.0,
    250, 0*NOISE_AMP*1.0                       // noise (G)
};
#endif

LIF::LIF(const char* name, HyPerCol * hc)
  : HyPerLayer(name, hc, MAX_CHANNELS)
{
   initialize(TypeLIFSimple);
}

LIF::LIF(const char* name, HyPerCol * hc, PVLayerType type)
  : HyPerLayer(name, hc, MAX_CHANNELS)
{
   initialize(type);
}

LIF::~LIF()
{
   if (numChannels > 0) {
      // conductances allocated contiguously so this frees all
      free(G_E);
   }
   free(Vth);
   free(rand_state);
}

int LIF::initialize(PVLayerType type)
{
   float time = 0.0f;
   int status = CL_SUCCESS;

   const size_t numNeurons = getNumNeurons();

   setParams(parent->parameters());
   clayer->layerType = type;

   G_E = G_I = G_IB = NULL;

   if (numChannels > 0) {
      G_E = (pvdata_t *) calloc(numNeurons*numChannels, sizeof(pvdata_t));
      assert(G_E != NULL);

      G_I  = G_E + 1*numNeurons;
      G_IB = G_E + 2*numNeurons;
   }

   // a random state variable is needed for every neuron/clthread
   rand_state = cl_random_init(numNeurons);

   // initialize layer data
   //
   Vth = (pvdata_t *) calloc(numNeurons, sizeof(pvdata_t));
   assert(Vth != NULL);
   for (size_t k = 0; k < numNeurons; k++){
      Vth[k] = VTH_REST;
   }

   parent->addLayer(this);

   if (parent->parameters()->value(name, "restart", 0) != 0) {
      readState(name, &time);
   }

   // initialize OpenCL parameters
   //
#ifdef PV_USE_OPENCL
   CLDevice * device = parent->getCLDevice();

   // TODO - fix to use device and layer parameters
   if (device->id() == 1) {
      nxl = 1;  nyl = 1;
   }
   else {
      nxl = 16; nyl = 16;
   }
   initializeThreadBuffers();
   initializeThreadKernels();
#endif

   return status;
}

#ifdef PV_USE_OPENCL
/**
 * Initialize OpenCL buffers.  This must be called after PVLayer data have
 * been allocated.
 */
int LIF::initializeThreadBuffers()
{
   int status = CL_SUCCESS;

   const size_t size    = getNumNeurons()  * sizeof(pvdata_t);
   const size_t size_ex = getNumExtended() * sizeof(pvdata_t);

   CLDevice * device = parent->getCLDevice();

   // these buffers are shared between host and device
   //

   // TODO - use constant memory
   clParams = device->createBuffer(CL_MEM_COPY_HOST_PTR, sizeof(lParams), &lParams);
   clRand   = device->createBuffer(CL_MEM_COPY_HOST_PTR, getNumNeurons()*sizeof(uint4), rand_state);

   clV    = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, clayer->V);
   clVth  = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, Vth);
   clG_E  = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, G_E);
   clG_I  = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, G_I);
   clG_IB = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, G_IB);

   clPhiE  = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, getChannel(CHANNEL_EXC));
   clPhiI  = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, getChannel(CHANNEL_INH));
   clPhiIB = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, getChannel(CHANNEL_INHB));

   clActivity = device->createBuffer(CL_MEM_COPY_HOST_PTR, size_ex, clayer->activity->data);
   clPrevTime = device->createBuffer(CL_MEM_COPY_HOST_PTR, size_ex, clayer->prevActivity);

   return status;
}

int LIF::initializeThreadKernels()
{
   int status = CL_SUCCESS;
   CLDevice * device = parent->getCLDevice();

   // create kernels
   //
   krUpdate = device->createKernel("kernels/LIF_update_state.cl",
                                   "LIF_update_state",
                                   "-I /Users/rasmussn/eclipse/workspace.petavision/PetaVisionII/src/kernels/");

   int argid = 0;

   status |= krUpdate->setKernelArg(argid++, 0.0f); // time (changed by updateState)
   status |= krUpdate->setKernelArg(argid++, 1.0f); // dt (changed by updateState)

   status |= krUpdate->setKernelArg(argid++, clayer->loc.nx);
   status |= krUpdate->setKernelArg(argid++, clayer->loc.ny);
   status |= krUpdate->setKernelArg(argid++, clayer->loc.nf);
   status |= krUpdate->setKernelArg(argid++, clayer->loc.nb);

   status |= krUpdate->setKernelArg(argid++, clParams);
   status |= krUpdate->setKernelArg(argid++, clRand);

   status |= krUpdate->setKernelArg(argid++, clV);
   status |= krUpdate->setKernelArg(argid++, clG_E);
   status |= krUpdate->setKernelArg(argid++, clG_I);
   status |= krUpdate->setKernelArg(argid++, clG_IB);
   status |= krUpdate->setKernelArg(argid++, clPhiI);
   status |= krUpdate->setKernelArg(argid++, clActivity);

   return status;
}
#endif

int LIF::setParams(PVParams * p)
{
   float dt_sec = .001 * parent->getDeltaTime();  // seconds

   clayer->params = &lParams;

   spikingFlag = (int) p->value(name, "spikingFlag", 1);

   lParams.Vrest = p->value(name, "Vrest", V_REST);
   lParams.Vexc  = p->value(name, "Vexc" , V_EXC);
   lParams.Vinh  = p->value(name, "Vinh" , V_INH);
   lParams.VinhB = p->value(name, "VinhB", V_INHB);

   lParams.tau   = p->value(name, "tau"  , TAU_VMEM);
   lParams.tauE  = p->value(name, "tauE" , TAU_EXC);
   lParams.tauI  = p->value(name, "tauI" , TAU_INH);
   lParams.tauIB = p->value(name, "tauIB", TAU_INHB);

   lParams.VthRest  = p->value(name, "VthRest" , VTH_REST);
   lParams.tauVth   = p->value(name, "tauVth"  , TAU_VTH);
   lParams.deltaVth = p->value(name, "deltaVth", DELTA_VTH);

   // NOTE: in LIFDefaultParams, noise ampE, ampI, ampIB were
   // ampE=0*NOISE_AMP*( 1.0/TAU_EXC )
   //       *(( TAU_INH * (V_REST-V_INH) + TAU_INHB * (V_REST-V_INHB) ) / (V_EXC-V_REST))
   // ampI=0*NOISE_AMP*1.0
   // ampIB=0*NOISE_AMP*1.0
   // 

   lParams.noiseAmpE  = p->value(name, "noiseAmpE" , 0.0f);
   lParams.noiseAmpI  = p->value(name, "noiseAmpI" , 0.0f);
   lParams.noiseAmpIB = p->value(name, "noiseAmpIB", 0.0f);

   lParams.noiseFreqE  = p->value(name, "noiseFreqE" , 250);
   lParams.noiseFreqI  = p->value(name, "noiseFreqI" , 250);
   lParams.noiseFreqIB = p->value(name, "noiseFreqIB", 250);
   
   if (dt_sec * lParams.noiseFreqE  > 1.0) lParams.noiseFreqE  = 1.0/dt_sec;
   if (dt_sec * lParams.noiseFreqI  > 1.0) lParams.noiseFreqI  = 1.0/dt_sec;
   if (dt_sec * lParams.noiseFreqIB > 1.0) lParams.noiseFreqIB = 1.0/dt_sec;

   return 0;
}

#ifdef PV_USE_OPENCL
int LIF::updateStateOpenCL(float time, float dt)
{
   int status = CL_SUCCESS;

   // setup and run kernel
   // a. unmap the state variables so device can read and write
   // b. pass state variables to kernel
   // c. run kernel
   // e. map the state variable for processing on CPU

   const int nx = clayer->loc.nx;
   const int ny = clayer->loc.ny;

   krUpdate->run(nx, ny, nxl, nyl, 0, NULL, &evList[0]);

   return status;
}
#endif

int LIF::updateState(float time, float dt)
{
   pv_debug_info("[%d]: LIF::updateState:", clayer->columnId);

#ifndef PV_USE_OPENCL
      const int nx = clayer->loc.nx;
      const int ny = clayer->loc.ny;
      const int nf = clayer->loc.nf;
      const int nb = clayer->loc.nb;

      pvdata_t * phiExc   = getChannel(CHANNEL_EXC);
      pvdata_t * phiInh   = getChannel(CHANNEL_INH);
      pvdata_t * phiInhB  = getChannel(CHANNEL_INHB);
      pvdata_t * activity = clayer->activity->data;

      if (spikingFlag == 1) {
         LIF_update_state(time, dt, nx, ny, nf, nb,
                          &lParams, rand_state,
                          clayer->V, clayer->Vth,
                          G_E, G_I, G_IB,
                          phiExc, phiInh, phiInhB, activity);

         // TODO - move to halo exchange so don't have to wait for data
         // calculate active indices
         //

         int numActive = 0;
         for (int k = 0; k < getNumNeurons(); k++) {
            const int kex = kIndexExtended(k, nx, ny, nf, nb);
            if (activity[kex] > 0.0) {
               clayer->activeIndices[numActive++] = globalIndexFromLocal(k, clayer->loc);
            }
            clayer->numActive = numActive;
         }
      }
#else
      return updateStateOpenCL(time, dt);
#endif

   return 0;
}

int LIF::writeState(const char * path, float time)
{
   HyPerLayer::writeState(path, time);

#ifdef DEBUG_OUTPUT
   // print activity at center of image

   int sx = clayer->numFeatures;
   int sy = sx*clayer->loc.nx;
   pvdata_t * a = clayer->activity->data;

   int n = (int) (sy*(clayer->loc.ny/2 - 1) + sx*(clayer->loc.nx/2));
   for (int f = 0; f < clayer->numFeatures; f++) {
      printf("f = %d, a[%d] = %f\n", f, n, a[n]);
      n += 1;
   }
   printf("\n");

   n = (int) (sy*(clayer->loc.ny/2 - 1) + sx*(clayer->loc.nx/2));
   n -= 8;
   for (int f = 0; f < clayer->numFeatures; f++) {
      printf("f = %d, a[%d] = %f\n", f, n, a[n]);
      n += 1;
   }
#endif

   return 0;
}

int LIF::findPostSynaptic(int dim, int maxSize, int col,
// input: which layer, which neuron
		HyPerLayer *lSource, float pos[],

		// output: how many of our neurons are connected.
		// an array with their indices.
		// an array with their feature vectors.
		int* nNeurons, int nConnectedNeurons[], float *vPos)
{
	return 0;
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
#  include "../kernels/LIF_update_state.cl"
#endif

#ifdef __cplusplus
}
#endif
