/*
 * Retina.cpp
 *
 *  Created on: Jul 29, 2008
 *
 */

#include "HyPerLayer.hpp"
#include "Retina.hpp"
#include "../columns/Random.hpp"
#include "../io/io.h"
#include "../include/default_params.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void Retina_spiking_update_state (
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
    Retina_params * params,
    taus_uint4 * rnd,
    float * GSynHead,
//    float * phiExc,
//    float * phiInh,
    float * activity,
    float * prevTime);

void Retina_nonspiking_update_state (
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
    Retina_params * params,
    float * GSynHead,
//    float * phiExc,
//    float * phiInh,
    float * activity);

#ifdef __cplusplus
}
#endif


namespace PV {

Retina::Retina() {
   initialize_base();
   // Default constructor to be called by derived classes.
   // It doesn't call Retina::initialize; instead, the derived class
   // should explicitly call Retina::initialize in its own initialization,
   // the way that Retina::initialize itself calls HyPerLayer::initialization.
   // This way, virtual methods called by initialize will be overridden
   // as expected.
}

Retina::Retina(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
//#ifdef PV_USE_OPENCL
//   if(gpuAccelerateFlag)
//      initializeGPU();
//#endif
}

Retina::~Retina()
{
   delete randState;
//#ifdef PV_USE_OPENCL
//   if((gpuAccelerateFlag)&&(spikingFlag)) {
//      delete clRand;
//   }
//   // Moved to HyPerLayer since evList is a HyPerLayer member variable
////    free(evList);
//#endif
}

int Retina::initialize_base() {
   numChannels = NUM_RETINA_CHANNELS;
   //for (int nbr=0; nbr<NUM_NEIGHBORHOOD; nbr++) {
   randState = NULL;
   //   rand_state_size[nbr] = 0UL;
   //   border_indices[nbr] = NULL;
   //}
   spikingFlag = true;
   rParams.abs_refractory_period = 0.0f;
   rParams.refractory_period = 0.0f;
   rParams.beginStim = 0.0f;
   rParams.endStim = -1.0;
   rParams.burstDuration = 1000.0;
   rParams.burstFreq = 1.0f;
   rParams.probBase = 0.0f;
   rParams.probStim = 1.0f;
   return PV_SUCCESS;
}

int Retina::initialize(const char * name, HyPerCol * hc) {
   int status = HyPerLayer::initialize(name, hc);

   setRetinaParams(parent->parameters());

//#ifdef PV_USE_OPENCL
//   numEvents=NUM_RETINA_EVENTS;
////this code was moved to Hyperlayer:initializeGPU():
////   CLDevice * device = parent->getCLDevice();
////
////   numWait = 0;
////   numEvents = NUM_RETINA_EVENTS;
////   evList = (cl_event *) malloc(numEvents*sizeof(cl_event));
////   assert(evList != NULL);
////
////   // TODO - fix to use device and layer parameters
////   if (device->id() == 1) {
////      nxl = 1;  nyl = 1;
////   }
////   else {
////      nxl = 16; nyl = 8;
////   }
////
////   const char * kernel_name;
////   if (spikingFlag) {
////      kernel_name = "Retina_spiking_update_state";
////   }
////   else {
////      kernel_name = "Retina_nonspiking_update_state";
////   }
////
////   initializeThreadBuffers(kernel_name);
////   initializeThreadKernels(kernel_name);
//#endif

   return status;
}

//#ifdef PV_USE_OPENCL
///**
// * Initialize OpenCL buffers.  This must be called after PVLayer data have
// * been allocated.
// */
//int Retina::initializeThreadBuffers(const char * kernel_name)
//{
//   int status = HyPerLayer::initializeThreadBuffers(kernel_name);
//
//   CLDevice * device = parent->getCLDevice();
//
//   clParams = device->createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(rParams), &rParams);
////   clParams->copyToDevice(&evUpdate);
////   status |= clWaitForEvents(1, &evUpdate);
////   clReleaseEvent(evUpdate);
//
//   if (spikingFlag) {
//      clRand   = device->createBuffer(CL_MEM_COPY_HOST_PTR, getNumNeurons()*sizeof(taus_uint4), *rand_state);
////      clRand->copyToDevice(&evUpdate);
////      status |= clWaitForEvents(1, &evUpdate);
////      clReleaseEvent(evUpdate);
//   }
//
//   return status;
//}
//
//int Retina::initializeThreadKernels(const char * kernel_name)
//{
//   char kernelPath[256];
//   char kernelFlags[256];
//
//   int status = CL_SUCCESS;
//   CLDevice * device = parent->getCLDevice();
//
//   const char * pvRelPath = "../PetaVision";
//   sprintf(kernelPath,  "%s/%s/src/kernels/Retina_update_state.cl", parent->getSrcPath(), pvRelPath);
//   sprintf(kernelFlags, "-D PV_USE_OPENCL -cl-fast-relaxed-math -I %s/%s/src/kernels/", parent->getSrcPath(), pvRelPath);
//
//   // create kernels
//   //
//   krUpdate = device->createKernel(kernelPath, kernel_name, kernelFlags);
////kernel name should already be set correctly!
////   if (spikingFlag) {
////      krUpdate = device->createKernel(kernelPath, kernel_name, kernelFlags);
////   }
////   else {
////      krUpdate = device->createKernel(kernelPath, "Retina_nonspiking_update_state", kernelFlags);
////   }
//
//   int argid = 0;
//
//   status |= krUpdate->setKernelArg(argid++, getNumNeurons());
//   status |= krUpdate->setKernelArg(argid++, parent->simulationTime());
//   status |= krUpdate->setKernelArg(argid++, parent->getDeltaTime());
//
//   status |= krUpdate->setKernelArg(argid++, clayer->loc.nx);
//   status |= krUpdate->setKernelArg(argid++, clayer->loc.ny);
//   status |= krUpdate->setKernelArg(argid++, clayer->loc.nf);
//   status |= krUpdate->setKernelArg(argid++, clayer->loc.nb);
//
//   status |= krUpdate->setKernelArg(argid++, clParams);
//   if (spikingFlag) {
//      status |= krUpdate->setKernelArg(argid++, clRand);
//   }
//
//   status |= krUpdate->setKernelArg(argid++, getChannelCLBuffer());
////   status |= krUpdate->setKernelArg(argid++, getChannelCLBuffer(CHANNEL_EXC));
////   status |= krUpdate->setKernelArg(argid++, getChannelCLBuffer(CHANNEL_INH));
//   status |= krUpdate->setKernelArg(argid++, clActivity);
//   if (spikingFlag) {
//      status |= krUpdate->setKernelArg(argid++, clPrevTime);
//   }
//
//   return status;
//}
//#endif

int Retina::communicateInitInfo() {
   int status = HyPerLayer::communicateInitInfo();
   if(parent->getNBatch() != 1){
      std::cout << "Retina does not support batches yet, TODO\n";
      exit(-1);
   }
   return status;
}

int Retina::allocateDataStructures() {
   int status = HyPerLayer::allocateDataStructures();

   assert(!parent->parameters()->presentAndNotBeenRead(name, "spikingFlag"));
   if (spikingFlag) {
      // // a random state variable is needed for every neuron/clthread
      //int numGlobalRNGs = getNumGlobalExtended();
      //unsigned int seed = parent->getObjectSeed(numGlobalRNGs);
      const PVLayerLoc * loc = getLayerLoc();
      //Allocate extended loc
      randState = new Random(parent, loc, true); // (taus_uint4 *) malloc(count * sizeof(taus_uint4));

      //unsigned int columnOffset = (unsigned int) kIndex(loc->kx0,loc->ky0,0,loc->nxGlobal,loc->nyGlobal,loc->nf);
      //allocateRandStateRestricted(loc->nx, loc->ny, loc->nf, seed+columnOffset, loc->nxGlobal * loc->nf);
      //int indexStride = (loc->nx+loc->halo.lt+loc->halo.rt)*loc->nf;
      //int globalExtStride = (loc->nxGlobal+loc->halo.dn+loc->halo.up)*loc->nf;
      //seed += getNumGlobalNeurons();
      //int indexBase = 0;
      //allocateRandStateBorder(NORTHWEST, loc->halo.lt, loc->halo.up, loc->nf, seed, globalExtStride, indexBase, indexStride);
      //allocateRandStateBorder(NORTH, loc->nx, loc->halo.up, loc->nf, seed + loc->halo.lt + loc->kx0, globalExtStride, indexBase+loc->halo.lt, indexStride);
      //allocateRandStateBorder(NORTHEAST, loc->halo.rt, loc->halo.up, loc->nf, seed + loc->halo.lt + loc->nxGlobal, globalExtStride, indexBase+loc->halo.lt+loc->nx, indexStride);
      //seed += (loc->nxGlobal + loc->halo.lt + loc->halo.rt) * loc->halo.up * loc->nf;
      //indexBase = kIndex(0, loc->halo.up, 0, loc->nx+loc->halo.lt+loc->halo.rt, loc->ny+loc->halo.dn+loc->halo.up, loc->nf);
      //allocateRandStateBorder(WEST, loc->halo.lt, loc->ny, loc->nf, seed + loc->ky0*(loc->halo.lt+loc->halo.rt)*loc->nf, (loc->halo.lt+loc->halo.rt)*loc->nf, indexBase, indexStride);
      //allocateRandStateBorder(EAST, loc->halo.rt, loc->ny, loc->nf, seed + loc->ky0*(loc->halo.lt+loc->halo.rt)*loc->nf, (loc->halo.lt+loc->halo.rt)*loc->nf, indexBase+loc->nx+loc->halo.lt, indexStride);
      //seed += (loc->halo.lt+loc->halo.rt) * loc->nyGlobal * loc->nf;
      //indexBase = kIndex(0, loc->halo.up+loc->ny, 0, loc->nx+loc->halo.lt+loc->halo.rt, loc->ny+loc->halo.dn+loc->halo.up, loc->nf);
      //allocateRandStateBorder(SOUTHWEST, loc->halo.lt, loc->halo.dn, loc->nf, seed, globalExtStride, indexBase, indexStride);
      //allocateRandStateBorder(SOUTH, loc->nx, loc->halo.dn, loc->nf, seed + loc->halo.lt + loc->kx0, globalExtStride, indexBase+loc->halo.lt, indexStride);
      //allocateRandStateBorder(SOUTHEAST, loc->halo.rt, loc->halo.up, loc->nf, seed + loc->halo.lt + loc->nxGlobal, globalExtStride, indexBase+loc->halo.lt+loc->nx, indexStride);
   }

   return status;
}

//int Retina::allocateRandStateRestricted(size_t xCount, size_t yCount, size_t fCount, unsigned int seedStart, unsigned int seedStride) {
//   int status = allocateRandState(0, xCount, yCount, fCount, seedStart, seedStride);
//   return status;
//}

//int Retina::allocateRandStateBorder(int neighbor, size_t xCount, size_t yCount, size_t fCount, unsigned int seedStart, unsigned int seedStride, int indexStart, int indexStride) {
//   assert(neighbor!=LOCAL && neighbor>=0 && neighbor<NUM_NEIGHBORHOOD);
//   int status = PV_SUCCESS;
//   if (!parent->icCommunicator()->hasNeighbor(neighbor)) {
//      if (status == PV_SUCCESS) status = allocateRandState(neighbor, xCount, yCount, fCount, seedStart, seedStride);
//      if (status == PV_SUCCESS) status = allocateBorderIndices(neighbor, xCount, yCount, fCount, indexStart, indexStride);
//   }
//   else {
//      parent->getObjectSeed(xCount*yCount*fCount); // Keep parent's random-seed management in sync across processes.
//   }
//   return status;
//}

//int Retina::allocateRandState(int neighbor, size_t xCount, size_t yCount, size_t fCount, unsigned int seedStart, unsigned int seedStride) {
//   assert(randState[neighbor]==NULL);
//   size_t count = xCount * yCount * fCount;
//   rand_state_size[neighbor] = count;
//   randState[neighbor] = new Random(parent, count); // (taus_uint4 *) malloc(count * sizeof(taus_uint4));
//   if (randState[neighbor]==NULL) {
//      fprintf(stderr, "Retina::allocateRandState error in rank %d.  Layer \"%s\" unable to create object of class Random.\n", parent->columnId(), getName());
//      exit(EXIT_FAILURE);
//   }
//
//   return PV_SUCCESS;
//}

//int Retina::allocateBorderIndices(int neighbor, size_t xCount, size_t yCount, size_t fCount, int indexStart, int indexStride) {
//   assert(neighbor!=LOCAL && neighbor>=0 && neighbor<NUM_NEIGHBORHOOD);
//   border_indices[neighbor] = (int *) malloc(rand_state_size[neighbor]*sizeof(int));
//   if (border_indices[neighbor]==NULL) {
//      fprintf(stderr, "%s \"%s\" error in rank %d process: setBorderIndices unable to allocate memory for neighbor %d\n.", getKeyword(), name, parent->columnId(), neighbor);
//      exit(EXIT_FAILURE);
//   }
//   int index = indexStart;
//   size_t kCount = xCount*fCount;
//   int j=0;
//   for (size_t y=0; y<yCount; y++) {
//      for (int k=0; k<kCount; k++) {
//         border_indices[neighbor][j++] = index+k;
//      }
//      index += indexStride;
//   }
//   return PV_SUCCESS;
//}

int Retina::allocateV() {
   clayer->V = NULL;
   return PV_SUCCESS;
}

int Retina::initializeV() {
   assert(getV() == NULL);
   return PV_SUCCESS;
}

int Retina::initializeActivity() {
   return updateState(parent->simulationTime(), parent->getDeltaTime());
}

int Retina::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerLayer::ioParamsFillGroup(ioFlag);
   ioParam_spikingFlag(ioFlag);
   ioParam_foregroundRate(ioFlag);
   ioParam_backgroundRate(ioFlag);
   ioParam_beginStim(ioFlag);
   ioParam_endStim(ioFlag);
   ioParam_burstFreq(ioFlag);
   ioParam_burstDuration(ioFlag);
   ioParam_refractoryPeriod(ioFlag);
   ioParam_absRefractoryPeriod(ioFlag);

   return status;
}

void Retina::ioParam_InitVType(enum ParamsIOFlag ioFlag) {
   parent->parameters()->handleUnnecessaryParameter(name, "InitVType");
   return;
}

void Retina::ioParam_spikingFlag(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "spikingFlag", &spikingFlag, true);
}

void Retina::ioParam_foregroundRate(enum ParamsIOFlag ioFlag) {
   PVParams * params = parent->parameters();
   if (ioFlag==PARAMS_IO_READ && !params->present(name, "foregroundRate")) {
      if (params->present(name, "noiseOnFreq")) {
         probStimParam = params->value(name, "noiseOnFreq");
         if (parent->columnId()==0) {
            fprintf(stderr, "Warning: noiseOnFreq is deprecated.  Use foregroundRate instead.\n");
         }
         return;
      }
      if (params->present(name, "poissonEdgeProb")) {
         probStimParam = params->value(name, "poissonEdgeProb");
         if (parent->columnId()==0) {
            fprintf(stderr, "Warning: poissonEdgeProb is deprecated.  Use foregroundRate instead.\n");
         }
         return;
      }
   }
   // noiseOnFreq and poissonEdgeProb were deprecated Jan 24, 2013
   // When support for noiseOnFreq and poissonEdgeProb is removed entirely, remove the above if-statement and keep the ioParamValue call below.
   parent->ioParamValue(ioFlag, name, "foregroundRate", &probStimParam, 1.0f);
}

void Retina::ioParam_backgroundRate(enum ParamsIOFlag ioFlag) {
   PVParams * params = parent->parameters();
   if (ioFlag==PARAMS_IO_READ && !params->present(name, "backgroundRate")) {
      if (params->present(name, "noiseOffFreq")) {
         probBaseParam = params->value(name, "noiseOffFreq");
         if (parent->columnId()==0) {
            fprintf(stderr, "Warning: noiseOffFreq is deprecated.  Use backgroundRate instead.\n");
         }
         return;
      }
      if (params->present(name, "poissonBlankProb")) {
         probBaseParam = params->value(name, "poissonBlankProb");
         if (parent->columnId()==0) {
            fprintf(stderr, "Warning: poissonEdgeProb is deprecated.  Use backgroundRate instead.\n");
         }
         return;
      }
   }
   // noiseOffFreq and poissonBlankProb was deprecated Jan 24, 2013
   // When support for noiseOnFreq and poissonEdgeProb is removed entirely, remove the above if-statement and keep the ioParamValue call below.
   parent->ioParamValue(ioFlag, name, "backgroundRate", &probBaseParam, 0.0f);
   if (ioFlag==PARAMS_IO_READ) {
      assert(!parent->parameters()->presentAndNotBeenRead(name, "foregroundRate"));
      if (probBaseParam > probStimParam) {
         fprintf(stderr, "Error in %s \"%s\": backgroundRate cannot be greater than foregroundRate.\n",
               getKeyword(), name);
         exit(EXIT_FAILURE);
      }
   }
}

void Retina::ioParam_beginStim(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "beginStim", &rParams.beginStim, 0.0);
}

void Retina::ioParam_endStim(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "endStim", &rParams.endStim, (double) FLT_MAX);
   if (ioFlag == PARAMS_IO_READ && rParams.endStim < 0) rParams.endStim = FLT_MAX;
}

void Retina::ioParam_burstFreq(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "burstFreq", &rParams.burstFreq, 1.0f);
}

void Retina::ioParam_burstDuration(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "burstDuration", &rParams.burstDuration, 1000.0f);
}

void Retina::ioParam_refractoryPeriod(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "spikingFlag"));
   if (spikingFlag){
      parent->ioParamValue(ioFlag, name, "refractoryPeriod", &rParams.refractory_period, (float) REFRACTORY_PERIOD);
   }
}

void Retina::ioParam_absRefractoryPeriod(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "spikingFlag"));
   if (spikingFlag){
      parent->ioParamValue(ioFlag, name, "absRefractoryPeriod", &rParams.abs_refractory_period, (float) ABS_REFRACTORY_PERIOD);
   }
}


int Retina::setRetinaParams(PVParams * p)
{
   clayer->params = &rParams;

   double dt_sec = parent->getDeltaTime() * .001;  // seconds
   float probStim = probStimParam * dt_sec;
   if (probStim > 1.0) probStim = 1.0f;
   float probBase = probBaseParam * dt_sec;
   if (probBase > 1.0) probBase = 1.0f;

   maxRate = probStim/dt_sec;

   // default parameters
   //
   rParams.probStim  = probStim;
   rParams.probBase  = probBase;

   return 0;
}


int Retina::readStateFromCheckpoint(const char * cpDir, double * timeptr) {
   int status = HyPerLayer::readStateFromCheckpoint(cpDir, timeptr);
   double filetime = 0.0;
   status = readRandStateFromCheckpoint(cpDir);
   return status;
}

int Retina::readRandStateFromCheckpoint(const char * cpDir) {
   int status = PV_SUCCESS;
   if (spikingFlag) {
      char * filename = parent->pathInCheckpoint(cpDir, getName(), "_rand_state.bin");
      status = readRandState(filename, parent->icCommunicator(), randState->getRNG(0), getLayerLoc(), true /*isExtended*/);
      free(filename);
   }
   return status;
}

int Retina::checkpointWrite(const char * cpDir) {
   int status = HyPerLayer::checkpointWrite(cpDir);

   // Save rand_state
   char filename[PV_PATH_MAX];
   int chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_rand_state.bin", cpDir, name);
   if(chars_needed >= PV_PATH_MAX) {
      if (parent->icCommunicator()->commRank()==0) {
         fprintf(stderr, "HyPerLayer::checkpointWrite error in layer \"%s\".  Base pathname \"%s/%s_rand_state.bin\" too long.\n", name, cpDir, name);
      }
      abort();
   }
   if (spikingFlag) {
      int rand_state_status = writeRandState(filename, parent->icCommunicator(), randState->getRNG(0), getLayerLoc(), true /*isExtended*/, parent->getVerifyWrites());
      if (rand_state_status != PV_SUCCESS) status = rand_state_status;
   }
   return status;
}


int Retina::updateStateOpenCL(double time, double dt)
{
   int status = CL_SUCCESS;

//#ifdef PV_USE_OPENCL
//   // wait for memory to be copied to device
//   if (numWait > 0) {
//       status |= clWaitForEvents(numWait, evList);
//   }
//   for (int i = 0; i < numWait; i++) {
//      clReleaseEvent(evList[i]);
//   }
//   numWait = 0;
//
//   status |= krUpdate->setKernelArg(1, time);
//   status |= krUpdate->setKernelArg(2, dt);
//   status |= krUpdate->run(getNumNeurons(), nxl*nyl, 0, NULL, &evUpdate);
//   krUpdate->finish();
//
//   status |= getChannelCLBuffer()->copyFromDevice(1, &evUpdate, &evList[getEVGSyn()]);
////   status |= getChannelCLBuffer(CHANNEL_EXC)->copyFromDevice(1, &evUpdate, &evList[getEVGSynE()]);
////   status |= getChannelCLBuffer(CHANNEL_INH)->copyFromDevice(1, &evUpdate, &evList[getEVGSynI()]);
//   status |= clActivity->copyFromDevice(1, &evUpdate, &evList[getEVActivity()]);
//   numWait += 2;
//
//#if PV_CL_COPY_BUFFERS
//   status |= clPhiE    ->copyFromDevice(1, &evUpdate, &evList[EV_R_PHI_E]);
//   status |= clPhiI    ->copyFromDevice(1, &evUpdate, &evList[EV_R_PHI_I]);
//   status |= clActivity->copyFromDevice(1, &evUpdate, &evList[EV_R_ACTIVITY]);
//   numWait += 3;
//#endif
//#endif

   return status;
}

int Retina::waitOnPublish(InterColComm* comm)
{
   // HyPerLayer::waitOnPublish already has a publish timer so don't duplicate
   int status = HyPerLayer::waitOnPublish(comm);

   // copy activity to device
   //
//#ifdef PV_USE_OPENCL
//#if PV_CL_COPY_BUFFERS
//   publish_timer->start();
//
//   status |= clActivity->copyToDevice(&evList[EV_R_ACTIVITY]);
//   numWait += 1;
//
//   publish_timer->stop();
//#endif
//#endif

   return status;
}

//! Updates the state of the Retina
/*!
 * REMARKS:
 *      - prevActivity[] buffer holds the time when a neuron last spiked.
 *      - not used if nonspiking
 *      - it sets the probStim and probBase.
 *              - probStim = noiseOnFreq * dt_sec * (phiExc - phiInh); the last ()  is V[k];
 *              - probBase = noiseOffFreq * dt_sec;
 *              .
 *      - activity[] is set to 0 or 1 depending on the return of spike()
 *      - this depends on the last time a neuron spiked as well as on V[]
 *      at the location of the neuron. This V[] is set by calling updateImage().
 *      - V points to the same memory space as data in the Image so that when Image
 *      is updated, V gets updated too.
 *      .
 *      .
 *
 *
 */
int Retina::updateState(double timed, double dt)
{
   //update_timer->start();
//#ifdef PV_USE_OPENCL
//   if((gpuAccelerateFlag)&&(true)) {
//      updateStateOpenCL(timed, dt);
//   }
//   else {
//#endif // PV_USE_OPENCL
      const int nx = clayer->loc.nx;
      const int ny = clayer->loc.ny;
      const int nf = clayer->loc.nf;
      const int nbatch = clayer->loc.nbatch;
      const PVHalo * halo = &clayer->loc.halo;

      pvdata_t * GSynHead   = GSyn[0];
      pvdata_t * activity = clayer->activity->data;

      if (spikingFlag == 1) {
         Retina_spiking_update_state(nbatch, getNumNeurons(), timed, dt, nx, ny, nf,
                                     halo->lt, halo->rt, halo->dn, halo->up,
                                     &rParams, randState->getRNG(0),
                                     GSynHead, activity, clayer->prevActivity);
      }
      else {
         Retina_nonspiking_update_state(nbatch, getNumNeurons(), timed, dt, nx, ny, nf,
                                        halo->lt, halo->rt, halo->dn, halo->up,
                                        &rParams, GSynHead, activity);
      }
//#ifdef PV_USE_OPENCL
//   }
//#endif // PV_USE_OPENCL

#ifdef DEBUG_PRINT
   char filename[132];
   sprintf(filename, "r_%d.tiff", (int)(2*timed));
   this->writeActivity(filename, timed);

   printf("----------------\n");
   for (int k = 0; k < 6; k++) {
      printf("host:: k==%d h_exc==%f h_inh==%f\n", k, phiExc[k], phiInh[k]);
   }
   printf("----------------\n");

#endif // DEBUG_PRINT
   //update_timer->stop();
   return 0;
}

int Retina::updateBorder(double time, double dt)
{
   // wait for OpenCL data transfers to finish
   HyPerLayer::updateBorder(time, dt);

   //unsigned int probBaseUInt = (unsigned int) floor(rParams.probBase * (float) Random::randomUIntMax());
   //for (int nbr=1; nbr<NUM_NEIGHBORHOOD; nbr++) {
   //   size_t sz = rand_state_size[nbr];
   //   assert(sz==0 || randState[nbr]!=NULL);
   //   for (size_t n=0; n<sz; n++) {
   //      getActivity()[border_indices[nbr][n]] = (randState[nbr]->randomUInt(n) < probBaseUInt) ? 1.0 : 0.0;
   //   }
   //}
   return PV_SUCCESS;
}

int Retina::outputState(double time, bool last)
{
   // io_timer->start();
   // io_timer->stop();

   // HyPerLayer::outputState already has an io timer so don't duplicate
   return HyPerLayer::outputState(time, last);
}

BaseObject * createRetina(char const * name, HyPerCol * hc) {
   return hc ? new Retina(name, hc) : NULL;
}

} // namespace PV

///////////////////////////////////////////////////////
//
// implementation of Retina kernels
//

#ifdef __cplusplus
extern "C" {
#endif

//#ifndef PV_USE_OPENCL
//#  include "../kernels/Retina_update_state.cl"
//#  include "../kernels/Retina_update_state.c"
//#endif
#ifndef PV_USE_OPENCL
#  include "../kernels/Retina_update_state.cl"
#else
#  undef PV_USE_OPENCL
#  include "../kernels/Retina_update_state.cl"
#  define PV_USE_OPENCL
#endif

#ifdef __cplusplus
}
#endif
