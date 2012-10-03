/*
 * LCALifLayer.cpp
 *
 *  Created on: Jun 26, 2012
 *      Author: slundquist
 */

#include "LCALIFLayer.hpp"
//#include "../kernels/LIF_update_state.cl"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif
void LCALIF_update_state(
   const int numNeurons,
   const float time,
   const float dt,

   const int nx,
   const int ny,
   const int nf,
   const int nb,

   const float tau_LCA,
   const float tau_thr,
   const float targetRate,

   pvdata_t * integratedSpikeCount,

   CL_MEM_CONST LIF_params * params,
   CL_MEM_GLOBAL uint4 * rnd,
   CL_MEM_GLOBAL float * V,
   CL_MEM_GLOBAL float * Vth,
   CL_MEM_GLOBAL float * G_E,
   CL_MEM_GLOBAL float * G_I,
   CL_MEM_GLOBAL float * G_IB,
   CL_MEM_GLOBAL float * GSynHead,
//    CL_MEM_GLOBAL float * GSynExc,
//    CL_MEM_GLOBAL float * GSynInh,
//    CL_MEM_GLOBAL float * GSynInhB,
//    CL_MEM_GLOBAL float * GSynGap,
   CL_MEM_GLOBAL float * activity,

   const float sum_gap,
   CL_MEM_GLOBAL float * G_Gap
);

#ifdef __cplusplus
}
#endif

namespace PV {
LCALIFLayer::LCALIFLayer() {
   initialize_base();
  // initialize(arguments) should *not* be called by the protected constructor.
}

LCALIFLayer::LCALIFLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc, MAX_CHANNELS + 1, "LCALIF_update_state");
}

int LCALIFLayer::initialize_base(){
   tau_LCA = 2000;
   tau_thr = 20000;
   targetRate = 1;
   integratedSpikeCount = NULL;
   return PV_SUCCESS;
}

int LCALIFLayer::initialize(const char * name, HyPerCol * hc, int num_channels, const char * kernel_name){
   LIF::initialize(name, hc, TypeLCA, num_channels, kernel_name);
   PVParams * params = hc->parameters();
   tau_LCA = params->value(name, "tau_LCA", tau_LCA);
   tau_thr = params->value(name, "tau_thr", tau_thr);
   targetRate = params->value(name, "targetRate", targetRate);
   return PV_SUCCESS;
}

LCALIFLayer::~LCALIFLayer()
{
   free(integratedSpikeCount);
}

int LCALIFLayer::allocateBuffers() {
   const size_t numNeurons = getNumNeurons();
   integratedSpikeCount = (pvdata_t *) calloc(numNeurons, sizeof(pvdata_t));
   assert(integratedSpikeCount != NULL);
   return LIF::allocateBuffers();
}

int LCALIFLayer::updateState(float time, float dt)
{
   LCALIF_update_state(getNumNeurons(), time, dt, clayer->loc.nx, clayer->loc.ny, clayer->loc.nf,
         clayer->loc.nb, tau_LCA, tau_thr, targetRate, integratedSpikeCount, &lParams,
         rand_state, clayer->V, Vth, G_E, G_I, G_IB, GSyn[0], clayer->activity->data, sumGap, G_Gap);
   return PV_SUCCESS;
}

/*
   int status = 0;
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
//      pvdata_t * GSynExc   = getChannel(CHANNEL_EXC);
//      pvdata_t * GSynInh   = getChannel(CHANNEL_INH);
//      pvdata_t * GSynInhB  = getChannel(CHANNEL_INHB);
      pvdata_t * activity = clayer->activity->data;

      switch (method) {
      case 'b':
         LIF_update_state_beginning(getNumNeurons(), time, dt, nx, ny, nf, nb, &lParams, rand_state, clayer->V, Vth,
               G_E, G_I, G_IB, GSynHead, activity);
         break;
      case 'o':
         LIF_update_state_original(getNumNeurons(), time, dt, nx, ny, nf, nb, &lParams, rand_state, clayer->V, Vth,
               G_E, G_I, G_IB, GSynHead, activity);
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
*/
//////////////////////////////////////////////////////
//
// implementation of LIF kernels
//

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PV_USE_OPENCL
#  include "../kernels/LCALIF_update_state.cl"
#else
#  undef PV_USE_OPENCL
#  undef USE_CLRANDOM
#  include "../kernels/LCALIF_update_state.cl"
#  define PV_USE_OPENCL
#endif

#ifdef __cplusplus
}
#endif

}




