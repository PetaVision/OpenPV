/*
 * LCALifLayer.cpp
 *
 *  Created on: Jun 26, 2012
 *      Author: slundquist & dpaiton
 */

#include "LCALIFLayer.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

//////////////////////////////////////////////////////
// implementation of LIF kernels
//////////////////////////////////////////////////////
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

//Kernel update state implementation receives all necessary variables
//for required computation. File is included above.
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

   pvdata_t dynVthScale,
   pvdata_t * dynVthRest,
   const float tauLCA,
   const float tauTHR,
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
   tauLCA = 200;
   tauTHR = 1000;
   targetRate = 50;
   dynVthScale = DEFAULT_DYNVTHSCALE;
   dynVthRest = NULL;
   integratedSpikeCount = NULL;
   return PV_SUCCESS;
}

int LCALIFLayer::initialize(const char * name, HyPerCol * hc, int num_channels, const char * kernel_name){
   LIFGap::initialize(name, hc, TypeLCA, num_channels, kernel_name);
   PVParams * params = hc->parameters();

   tauLCA     = params->value(name, "tauLCA", tauLCA);
   tauTHR     = params->value(name, "tauTHR", tauTHR);
   targetRate = params->value(name, "targetRate", targetRate);

   //Initialize dynVthRest to vthRest
   for (int i = 0; i < (int)getNumNeurons(); i++){
      //std::cout << "VthRest: " << VthRest << "\n";
      dynVthRest[i] = lParams.VthRest;
   }
   float defaultDynVthScale = lParams.VthRest-lParams.Vrest;
   dynVthScale = defaultDynVthScale > 0 ? dynVthScale : DEFAULT_DYNVTHSCALE;
   dynVthScale = params->value(name, "dynVthScale", dynVthScale);
   if (dynVthScale <= 0) {
      if (hc->columnId()==0) {
         fprintf(stderr,"LCALIFLayer \"%s\": dynVthScale must be positive (value in params is %f).\n", name, dynVthScale);
      }
      abort();
   }

   return PV_SUCCESS;
}

LCALIFLayer::~LCALIFLayer()
{
   free(integratedSpikeCount);
   free(dynVthRest);
}

int LCALIFLayer::allocateBuffers() {
   const size_t numNeurons = getNumNeurons();
   //Allocate data to keep track of trace
   integratedSpikeCount = (pvdata_t *) calloc(numNeurons, sizeof(pvdata_t));
   assert(integratedSpikeCount != NULL);
   dynVthRest = (pvdata_t *) calloc(numNeurons, sizeof(pvdata_t));
   assert(dynVthRest != NULL);
   return LIFGap::allocateBuffers();
}

int LCALIFLayer::updateState(float time, float dt)
{
   //Call update_state kernel
//   std::cout << clayer->activity->data[1000] << " " << integratedSpikeCount[1000] << "\n";
   LCALIF_update_state(getNumNeurons(), time, dt, clayer->loc.nx, clayer->loc.ny, clayer->loc.nf,
         clayer->loc.nb, dynVthScale, dynVthRest, tauLCA, tauTHR, targetRate, integratedSpikeCount, &lParams,
         rand_state, clayer->V, Vth, G_E, G_I, G_IB, GSyn[0], clayer->activity->data, sumGap, G_Gap);
   return PV_SUCCESS;
}
}
