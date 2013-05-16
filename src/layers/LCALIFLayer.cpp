/*
 * LCALIFLayer.cpp
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
   const double timed,
   const double dt,

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
   CL_MEM_GLOBAL float * Vattained,
   CL_MEM_GLOBAL float * Vmeminf,
   const int normalizeInputFlag,
   CL_MEM_GLOBAL float * GSynExcEffective,
   CL_MEM_GLOBAL float * GSynInhEffective,
   CL_MEM_GLOBAL float * excitatoryNoise,
   CL_MEM_GLOBAL float * inhibitoryNoise,
   CL_MEM_GLOBAL float * inhibNoiseB
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
   initialize(name, hc, MAX_CHANNELS + 2, "LCALIF_update_state");
}

int LCALIFLayer::initialize_base(){
   tauTHR = 1000;
   targetRateHz = 1;
   Vscale = DEFAULT_DYNVTHSCALE;
   Vadpt = NULL;
   integratedSpikeCount = NULL;
   G_Norm = NULL;
   GSynExcEffective = NULL;
   normalizeInputFlag = false;
   return PV_SUCCESS;
}

int LCALIFLayer::initialize(const char * name, HyPerCol * hc, int num_channels, const char * kernel_name){
   LIFGap::initialize(name, hc, TypeLCA, num_channels, kernel_name);
   PVParams * params = hc->parameters();

   tauTHR     = params->value(name, "tauTHR", tauTHR);
   targetRateHz = params->value(name, "targetRate", targetRateHz);
   normalizeInputFlag = params->value(name, "normalizeInput", (double) normalizeInputFlag);

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
   for (int k=0; k<numNeurons; k++) { 
      integratedSpikeCount[k] = targetRateHz/1000; // Initialize integrated spikes to non-zero value
      Vadpt[k]                = lParams.VthRest;   // Initialize integrated spikes to non-zero value
      Vattained[k]            = lParams.Vrest;
      Vmeminf[k]              = lParams.Vrest;
   }

   return PV_SUCCESS;
}

LCALIFLayer::~LCALIFLayer()
{
   free(integratedSpikeCount); integratedSpikeCount = NULL;
   free(Vadpt); Vadpt = NULL;
   free(Vattained); Vattained = NULL;
   free(Vmeminf); Vmeminf = NULL;
}

int LCALIFLayer::allocateBuffers() {
   const size_t numNeurons = getNumNeurons();
   //Allocate data to keep track of trace
   integratedSpikeCount = (pvdata_t *) calloc(numNeurons, sizeof(pvdata_t));
   assert(integratedSpikeCount != NULL);
   Vadpt = (pvdata_t *) calloc(numNeurons, sizeof(pvdata_t));
   assert(Vadpt != NULL);
   Vattained = (pvdata_t *) calloc(numNeurons, sizeof(pvdata_t));
   assert(Vattained != NULL);
   Vmeminf = (pvdata_t *) calloc(numNeurons, sizeof(pvdata_t));
   assert(Vattained != NULL);
   G_Norm = (pvdata_t *) calloc(numNeurons, sizeof(pvdata_t));
   assert(G_Norm != NULL);
   GSynExcEffective = (pvdata_t *) calloc(numNeurons, sizeof(pvdata_t));
   assert(GSynExcEffective != NULL);
   GSynInhEffective = (pvdata_t *) calloc(numNeurons, sizeof(pvdata_t));
   assert(GSynInhEffective != NULL);
   excitatoryNoise = (pvdata_t *) calloc(numNeurons, sizeof(pvdata_t));
   assert(excitatoryNoise != NULL);
   inhibitoryNoise = (pvdata_t *) calloc(numNeurons, sizeof(pvdata_t));
   assert(inhibitoryNoise != NULL);
   inhibNoiseB = (pvdata_t *) calloc(numNeurons, sizeof(pvdata_t));
   assert(inhibNoiseB != NULL);
   return LIFGap::allocateBuffers();
}

int LCALIFLayer::updateState(double timed, double dt)
{
   //Calculate_state kernel
   for (int k=0; k<getNumNeurons(); k++) {
      G_Norm[k] = GSyn[CHANNEL_NORM][k]; // Copy GSyn buffer on normalizing channel for checkpointing, since LCALIF_update_state will blank the GSyn's
   }
   LCALIF_update_state(getNumNeurons(), timed, dt, clayer->loc.nx, clayer->loc.ny, clayer->loc.nf,
         clayer->loc.nb, Vscale, Vadpt, tauTHR, targetRateHz, integratedSpikeCount, &lParams,
         rand_state, clayer->V, Vth, G_E, G_I, G_IB, GSyn[0], clayer->activity->data, sumGap, Vattained, Vmeminf, (int) normalizeInputFlag,
         GSynExcEffective, GSynInhEffective, excitatoryNoise, inhibitoryNoise, inhibNoiseB);
   updateActiveIndices();
   return PV_SUCCESS;
}

int LCALIFLayer::checkpointRead(const char * cpDir, double * timef) {
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
   readBufferFile(filename, icComm, &timed, &integratedSpikeCount, 1, /*extended*/ false, getLayerLoc());
   if( (float) timed != *timef && parent->icCommunicator()->commRank() == 0 ) {
      fprintf(stderr, "Warning: %s and %s_A.pvp have different timestamps: %f versus %f\n", filename, name, (float) timed, *timef);
   }

   chars_needed = snprintf(filename, PV_PATH_MAX, "%s_Vadpt.pvp", basepath);
      assert(chars_needed < PV_PATH_MAX);
      readBufferFile(filename, icComm, &timed, &Vadpt, 1, /*extended*/ false, getLayerLoc());
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
   writeBufferFile(filename, icComm, timed, &integratedSpikeCount, 1, /*extended*/false, getLayerLoc());

   chars_needed = snprintf(filename, PV_PATH_MAX, "%s_Vadpt.pvp", basepath);
   assert(chars_needed < PV_PATH_MAX);
   writeBufferFile(filename, icComm, timed, &Vadpt, 1, /*extended*/false, getLayerLoc());

   chars_needed = snprintf(filename, PV_PATH_MAX, "%s_Vattained.pvp", basepath);
   assert(chars_needed < PV_PATH_MAX);
   writeBufferFile(filename, icComm, timed, &Vattained, 1, /*extended*/false, getLayerLoc());

   chars_needed = snprintf(filename, PV_PATH_MAX, "%s_Vmeminf.pvp", basepath);
   assert(chars_needed < PV_PATH_MAX);
   writeBufferFile(filename, icComm, timed, &Vmeminf, 1, /*extended*/false, getLayerLoc());

   chars_needed = snprintf(filename, PV_PATH_MAX, "%s_G_Norm.pvp", basepath);
   assert(chars_needed < PV_PATH_MAX);
   writeBufferFile(filename, icComm, timed, &G_Norm, 1, /*extended*/false, getLayerLoc());

   chars_needed = snprintf(filename, PV_PATH_MAX, "%s_GSynExcEffective.pvp", basepath);
   assert(chars_needed < PV_PATH_MAX);
   writeBufferFile(filename, icComm, timed, &GSynExcEffective, 1, /*extended*/false, getLayerLoc());

   chars_needed = snprintf(filename, PV_PATH_MAX, "%s_GSynInhEffective.pvp", basepath);
   assert(chars_needed < PV_PATH_MAX);
   writeBufferFile(filename, icComm, timed, &GSynInhEffective, 1, /*extended*/false, getLayerLoc());

   chars_needed = snprintf(filename, PV_PATH_MAX, "%s_excitatoryNoise.pvp", basepath);
   assert(chars_needed < PV_PATH_MAX);
   writeBufferFile(filename, icComm, timed, &excitatoryNoise, 1, /*extended*/false, getLayerLoc());

   chars_needed = snprintf(filename, PV_PATH_MAX, "%s_inhibitoryNoise.pvp", basepath);
   assert(chars_needed < PV_PATH_MAX);
   writeBufferFile(filename, icComm, timed, &inhibitoryNoise, 1, /*extended*/false, getLayerLoc());

   chars_needed = snprintf(filename, PV_PATH_MAX, "%s_inhibNoiseB.pvp", basepath);
   assert(chars_needed < PV_PATH_MAX);
   writeBufferFile(filename, icComm, timed, &inhibNoiseB, 1, /*extended*/false, getLayerLoc());

   return status;
}


}  // namespace PV
