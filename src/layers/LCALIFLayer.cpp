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

   pvdata_t Vscale,
   pvdata_t * Vadpt,
   const float tauTHR,
   const float targetRateHz,

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
   tauTHR = 1000;
   targetRateHz = 1;
   Vscale = DEFAULT_DYNVTHSCALE;
   Vadpt = NULL;
   integratedSpikeCount = NULL;
   return PV_SUCCESS;
}

int LCALIFLayer::initialize(const char * name, HyPerCol * hc, int num_channels, const char * kernel_name){
   LIFGap::initialize(name, hc, TypeLCA, num_channels, kernel_name);
   PVParams * params = hc->parameters();

   tauTHR     = params->value(name, "tauTHR", tauTHR);
   targetRateHz = params->value(name, "targetRate", targetRateHz);

   float defaultDynVthScale = lParams.VthRest-lParams.Vrest;
   Vscale = defaultDynVthScale > 0 ? defaultDynVthScale : DEFAULT_DYNVTHSCALE;
   Vscale = params->value(name, "Vscale", Vscale);
   if (Vscale <= 0) {
      if (hc->columnId()==0) {
         fprintf(stderr,"LCALIFLayer \"%s\": Vscale must be positive (value in params is %f).\n", name, Vscale);
      }
      abort();
   }

   int numNeurons = getNumNeurons();
   for (int k=0; k<numNeurons; k++) { // Initialize integrated spikes to non-zero value
      integratedSpikeCount[k] = targetRateHz/1000;
   }

   for (int k=0; k<numNeurons; k++) { // Initialize integrated spikes to non-zero value
      Vadpt[k] = lParams.VthRest;
   }

   return PV_SUCCESS;
}

LCALIFLayer::~LCALIFLayer()
{
   free(integratedSpikeCount);
   free(Vadpt);
}

int LCALIFLayer::allocateBuffers() {
   const size_t numNeurons = getNumNeurons();
   //Allocate data to keep track of trace
   integratedSpikeCount = (pvdata_t *) calloc(numNeurons, sizeof(pvdata_t));
   assert(integratedSpikeCount != NULL);
   Vadpt = (pvdata_t *) calloc(numNeurons, sizeof(pvdata_t));
   assert(Vadpt != NULL);
   return LIFGap::allocateBuffers();
}

int LCALIFLayer::updateState(float time, float dt)
{
   //Calculate_state kernel
//   std::cout << clayer->activity->data[1000] << " " << integratedSpikeCount[1000] << "\n";
   LCALIF_update_state(getNumNeurons(), time, dt, clayer->loc.nx, clayer->loc.ny, clayer->loc.nf,
         clayer->loc.nb, Vscale, Vadpt, tauTHR, targetRateHz, integratedSpikeCount, &lParams,
         rand_state, clayer->V, Vth, G_E, G_I, G_IB, GSyn[0], clayer->activity->data, sumGap, G_Gap);

//   int nx = clayer->loc.nx;
//   int ny = clayer->loc.ny;
//   int nb = clayer->loc.nb;
//   int nf = clayer->loc.nf;
//
//   for(int k = 0; k < nx*ny*nf; k++){
//      int kex = kIndexExtended(k, nx, ny, nf, nb);
//      if (clayer->activity->data[kex] == 1){
//         int xPos = kxPos(k, nx, ny, nf);
//         int yPos = kyPos(k, nx, ny, nf);
//         std::cout << "After Extended Index: " << kex << "   Rest Coords: (" << xPos << "," << yPos <<")\n";
//      }
//   }


   updateActiveIndices();
   return PV_SUCCESS;
}

int LCALIFLayer::checkpointRead(const char * cpDir, float * timef) {
   int status = LIFGap::checkpointRead(cpDir, timef);
   InterColComm * icComm = parent->icCommunicator();
   char basepath[PV_PATH_MAX];
   char filename[PV_PATH_MAX];
   int lenbase = snprintf(basepath, PV_PATH_MAX, "%s/%s", cpDir, name);
   if (lenbase+strlen("_integratedSpikeCount.pvp") >= PV_PATH_MAX) { // currently _integratedSpikeCount.pvp is the longest suffix needed
      if (icComm->commRank()==0) {
         fprintf(stderr, "HyPerLayer::checkpointRead error in layer \"%s\".  Base pathname \"%s/%s_\" too long.\n", name, cpDir, name);
      }
      abort();
   }
   double timed;
   assert(filename != NULL);
   int chars_needed = snprintf(filename, PV_PATH_MAX, "%s_integratedSpikeCount.pvp", basepath);
   assert(chars_needed < PV_PATH_MAX);
   readBufferFile(filename, icComm, &timed, integratedSpikeCount, 1, /*extended*/ false, /*contiguous*/false, getLayerLoc());
   if( (float) timed != *timef && parent->icCommunicator()->commRank() == 0 ) {
      fprintf(stderr, "Warning: %s and %s_A.pvp have different timestamps: %f versus %f\n", filename, name, (float) timed, *timef);
   }

   chars_needed = snprintf(filename, PV_PATH_MAX, "%s_Vadpt.pvp", basepath);
      assert(chars_needed < PV_PATH_MAX);
      readBufferFile(filename, icComm, &timed, Vadpt, 1, /*extended*/ false, /*contiguous*/false, getLayerLoc());
      if( (float) timed != *timef && parent->icCommunicator()->commRank() == 0 ) {
         fprintf(stderr, "Warning: %s and %s_A.pvp have different timestamps: %f versus %f\n", filename, name, (float) timed, *timef);
      }

   return status;
}

int LCALIFLayer::checkpointWrite(const char * cpDir) {
   int status = LIFGap::checkpointWrite(cpDir);
   InterColComm * icComm = parent->icCommunicator();
   char basepath[PV_PATH_MAX];
   char filename[PV_PATH_MAX];
   int lenbase = snprintf(basepath, PV_PATH_MAX, "%s/%s", cpDir, name);
   if (lenbase+strlen("_integratedSpikeCount.pvp") >= PV_PATH_MAX) { // currently _integratedSpikeCount.pvp is the longest suffix needed
      if (icComm->commRank()==0) {
         fprintf(stderr, "LCALIFLayer::checkpointWrite error in layer \"%s\".  Base pathname \"%s/%s_\" too long.\n", name, cpDir, name);
      }
      abort();
   }
   double timed = (double) parent->simulationTime();
   int chars_needed = snprintf(filename, PV_PATH_MAX, "%s_integratedSpikeCount.pvp", basepath);
   assert(chars_needed < PV_PATH_MAX);
   writeBufferFile(filename, icComm, timed, integratedSpikeCount, 1, /*extended*/false, /*contiguous*/false, getLayerLoc());

   chars_needed = snprintf(filename, PV_PATH_MAX, "%s_Vadpt.pvp", basepath);
   assert(chars_needed < PV_PATH_MAX);
   writeBufferFile(filename, icComm, timed, Vadpt, 1, /*extended*/false, /*contiguous*/false, getLayerLoc());

   return status;
}


}  // namespace PV
