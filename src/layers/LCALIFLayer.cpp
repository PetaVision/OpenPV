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
//#ifndef PV_USE_OPENCL
//#  include "../kernels/LCALIF_update_state.cl"
//#else
#  undef PV_USE_OPENCL
#  include "../kernels/LCALIF_update_state.cl"
#  define PV_USE_OPENCL
//#endif
#ifdef __cplusplus
}
#endif

//Kernel update state implementation receives all necessary variables
//for required computation. File is included above.
#ifdef __cplusplus
extern "C" {
#endif
void LCALIF_update_state(
   const int nbatch,
   const int numNeurons,
   const double timed,
   const double dt,

   const int nx,
   const int ny,
   const int nf,
   const int lt,
   const int rt,
   const int dn,
   const int up,

   pvdata_t Vscale,
   pvdata_t * Vadpt,
   const float tauTHR,
   const float targetRateHz,

   pvdata_t * integratedSpikeCount,

   CL_MEM_CONST LIF_params * params,
   CL_MEM_GLOBAL taus_uint4 * rnd,
   CL_MEM_GLOBAL float * V,
   CL_MEM_GLOBAL float * Vth,
   CL_MEM_GLOBAL float * G_E,
   CL_MEM_GLOBAL float * G_I,
   CL_MEM_GLOBAL float * G_IB,
   CL_MEM_GLOBAL float * GSynHead,
   CL_MEM_GLOBAL float * activity,

   const pvgsyndata_t * gapStrength,
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
   initialize(name, hc, "LCALIF_update_state");
}

int LCALIFLayer::initialize_base(){
   numChannels = 5;
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

int LCALIFLayer::initialize(const char * name, HyPerCol * hc, const char * kernel_name){
   LIFGap::initialize(name, hc, kernel_name);
   PVParams * params = hc->parameters();

   float defaultDynVthScale = lParams.VthRest-lParams.Vrest;
   Vscale = defaultDynVthScale > 0 ? defaultDynVthScale : DEFAULT_DYNVTHSCALE;
   if (Vscale <= 0) {
      if (hc->columnId()==0) {
         fprintf(stderr,"LCALIFLayer \"%s\": Vscale must be positive (value in params is %f).\n", name, Vscale);
      }
      abort();
   }

   return PV_SUCCESS;
}

int LCALIFLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = LIFGap::ioParamsFillGroup(ioFlag);
   ioParam_tauTHR(ioFlag);
   ioParam_targetRate(ioFlag);
   ioParam_normalizeInput(ioFlag);
   ioParam_Vscale(ioFlag);
   return status;
}

void LCALIFLayer::ioParam_tauTHR(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "tauTHR", &tauTHR, tauTHR);
}

void LCALIFLayer::ioParam_targetRate(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "targetRate", &targetRateHz, targetRateHz);
}

void LCALIFLayer::ioParam_normalizeInput(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "normalizeInput", &normalizeInputFlag, normalizeInputFlag);
}

void LCALIFLayer::ioParam_Vscale(enum ParamsIOFlag ioFlag) {
   PVParams * p = parent->parameters();
   assert(!p->presentAndNotBeenRead(name, "VthRest"));
   assert(!p->presentAndNotBeenRead(name, "Vrest"));
   float defaultDynVthScale = lParams.VthRest-lParams.Vrest;
   Vscale = defaultDynVthScale > 0 ? defaultDynVthScale : DEFAULT_DYNVTHSCALE;
   parent->ioParamValue(ioFlag, name, "Vscale", &Vscale, Vscale);
}

LCALIFLayer::~LCALIFLayer()
{
   free(integratedSpikeCount); integratedSpikeCount = NULL;
   free(Vadpt); Vadpt = NULL;
   free(Vattained); Vattained = NULL;
   free(Vmeminf); Vmeminf = NULL;
}

int LCALIFLayer::allocateDataStructures() {
   int status = LIFGap::allocateDataStructures();

   int numNeurons = getNumNeuronsAllBatches();
   for (int k=0; k<numNeurons; k++) {
      integratedSpikeCount[k] = targetRateHz/1000; // Initialize integrated spikes to non-zero value
      Vadpt[k]                = lParams.VthRest;   // Initialize integrated spikes to non-zero value
      Vattained[k]            = lParams.Vrest;
      Vmeminf[k]              = lParams.Vrest;
   }

   return status;
}

int LCALIFLayer::allocateBuffers() {
   const size_t numNeurons = getNumNeuronsAllBatches();
   //Allocate data to keep track of trace
   int status = PV_SUCCESS;
   if (status==PV_SUCCESS) status = allocateBuffer(&integratedSpikeCount, numNeurons, "integratedSpikeCount");
   if (status==PV_SUCCESS) status = allocateBuffer(&Vadpt, numNeurons, "Vadpt");
   if (status==PV_SUCCESS) status = allocateBuffer(&Vattained, numNeurons, "Vattained");
   if (status==PV_SUCCESS) status = allocateBuffer(&Vmeminf, numNeurons, "Vmeminf");
   if (status==PV_SUCCESS) status = allocateBuffer(&G_Norm, numNeurons, "G_Norm");
   if (status==PV_SUCCESS) status = allocateBuffer(&GSynExcEffective, numNeurons, "GSynExcEffective");
   if (status==PV_SUCCESS) status = allocateBuffer(&GSynInhEffective, numNeurons, "GSynInhEffective");
   if (status==PV_SUCCESS) status = allocateBuffer(&excitatoryNoise, numNeurons, "excitatoryNoise");
   if (status==PV_SUCCESS) status = allocateBuffer(&inhibitoryNoise, numNeurons, "inhibitoryNoise");
   if (status==PV_SUCCESS) status = allocateBuffer(&inhibNoiseB, numNeurons, "inhibNoiseB");
   if (status!=PV_SUCCESS) exit(EXIT_FAILURE);
   return LIFGap::allocateBuffers();
}

int LCALIFLayer::updateState(double timed, double dt)
{
   //Calculate_state kernel
   for (int k=0; k<getNumNeuronsAllBatches(); k++) {
      G_Norm[k] = GSyn[CHANNEL_NORM][k]; // Copy GSyn buffer on normalizing channel for checkpointing, since LCALIF_update_state will blank the GSyn's
   }
   LCALIF_update_state(clayer->loc.nbatch, getNumNeurons(), timed, dt, clayer->loc.nx, clayer->loc.ny, clayer->loc.nf,
         clayer->loc.halo.lt, clayer->loc.halo.rt, clayer->loc.halo.dn, clayer->loc.halo.up, Vscale, Vadpt, tauTHR, targetRateHz, integratedSpikeCount, &lParams,
         randState->getRNG(0), clayer->V, Vth, G_E, G_I, G_IB, GSyn[0], clayer->activity->data, getGapStrength(), Vattained, Vmeminf, (int) normalizeInputFlag,
         GSynExcEffective, GSynInhEffective, excitatoryNoise, inhibitoryNoise, inhibNoiseB);
   return PV_SUCCESS;
}

int LCALIFLayer::readStateFromCheckpoint(const char * cpDir, double * timeptr) {
   int status = LIFGap::readStateFromCheckpoint(cpDir, timeptr);
   double filetime = 0.0;
   status = read_integratedSpikeCountFromCheckpoint(cpDir, timeptr);
   status = readVadptFromCheckpoint(cpDir, timeptr);
   return status;
}

int LCALIFLayer::read_integratedSpikeCountFromCheckpoint(const char * cpDir, double * timeptr) {
   char * filename = parent->pathInCheckpoint(cpDir, getName(), "_integratedSpikeCount.pvp");
   int status = readBufferFile(filename, parent->icCommunicator(), timeptr, &Vth, 1, /*extended*/true, getLayerLoc());
   assert(status==PV_SUCCESS);
   free(filename);
   return status;
}

int LCALIFLayer::readVadptFromCheckpoint(const char * cpDir, double * timeptr) {
   char * filename = parent->pathInCheckpoint(cpDir, getName(), "_Vadpt.pvp");
   int status = readBufferFile(filename, parent->icCommunicator(), timeptr, &Vth, 1, /*extended*/true, getLayerLoc());
   assert(status==PV_SUCCESS);
   free(filename);
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

BaseObject * createLCALIFLayer(char const * name, HyPerCol * hc) {
   return hc ? new LCALIFLayer(name, hc) : NULL;
}

}  // namespace PV
