/*
 * LIF.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: Craig Rasmussen
 */

#include "HyPerLayer.hpp"
#include "LIF.hpp"

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

void LIF_update_state_arma(
    const int nbatch,
    const int numNeurons,
    const float time,
    const float dt,

    const int nx,
    const int ny,
    const int nf,
    const int lt,
    const int rt,
    const int dn,
    const int up,

    LIF_params * params,
    taus_uint4 * rnd,

    float * V,
    float * Vth,
    float * G_E,
    float * G_I,
    float * G_IB,
    float * GSynHead,
    float * activity);

void LIF_update_state_beginning(
    const int nbatch,
    const int numNeurons,
    const float time,
    const float dt,

    const int nx,
    const int ny,
    const int nf,
    const int lt,
    const int rt,
    const int dn,
    const int up,

    LIF_params * params,
    taus_uint4 * rnd,

    float * V,
    float * Vth,
    float * G_E,
    float * G_I,
    float * G_IB,
    float * GSynHead,
    float * activity);

void LIF_update_state_original(
    const int nbatch,
    const int numNeurons,
    const float time,
    const float dt,

    const int nx,
    const int ny,
    const int nf,
    const int lt,
    const int rt,
    const int dn,
    const int up,

    LIF_params * params,
    taus_uint4 * rnd,

    float * V,
    float * Vth,
    float * G_E,
    float * G_I,
    float * G_IB,
    float * GSynHead,
    float * activity);

#ifdef __cplusplus
}
#endif

namespace PV
{

LIF::LIF() {
   initialize_base();
   // this is the constructor to be used by derived classes, it does not produce
   // a function class but is asking for an init by the derived class
}

LIF::LIF(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc, "LIF_update_state");
//#ifdef PV_USE_OPENCL
//   if(gpuAccelerateFlag)
//      initializeGPU();
//#endif
}

LIF::~LIF() {
   if (numChannels > 0) {
      // conductances allocated contiguously so this frees all
      free(G_E);
   }
   free(Vth);
   delete randState;
   free(methodString);

//#ifdef PV_USE_OPENCL
////hyperlayer is destroying these:
////   delete krUpdate;
////
////   free(evList);
////
////   delete clParams;
//   if(gpuAccelerateFlag) {
//      delete clRand;
//      delete clVth;
//      delete clG_E;
//      delete clG_I;
//      delete clG_IB;
//   }
//#endif

}

int LIF::initialize_base() {
   numChannels = 3;
   randState = NULL;
   Vth = NULL;
   G_E = NULL;
   G_I = NULL;
   G_IB = NULL;
   methodString = NULL;

//#ifdef PV_USE_OPENCL
//   clRand = NULL;
//   clVth = NULL;
//   clG_E = NULL;
//   clG_I = NULL;
//   clG_IB = NULL;
//#endif // PV_USE_OPEN_CL

   return PV_SUCCESS;
}

// Initialize this class
int LIF::initialize(const char * name, HyPerCol * hc, const char * kernel_name) {
   HyPerLayer::initialize(name, hc);
   clayer->params = &lParams;

//   // initialize OpenCL parameters
//   //
//   //This stuff is commented out for now, but will be used later and added to
//   //its own initializeGPU method
//#ifdef PV_USE_OPENCL
//   numEvents=NUM_LIF_EVENTS;
////   CLDevice * device = parent->getCLDevice();
////
////   // TODO - fix to use device and layer parameters
////   if (device->id() == 1) {
////      nxl = 1;  nyl = 1;
////   }
////   else {
////      nxl = 16; nyl = 8;
////   }
////
////   numWait = 0;
////   numEvents = getNumCLEvents(); //NUM_LIF_EVENTS;
////   evList = (cl_event *) malloc(numEvents*sizeof(cl_event));
////   assert(evList != NULL);
////
////   numKernelArgs = 0;
////   initializeThreadBuffers(kernel_name);
////   initializeThreadKernels(kernel_name);
////
////   DataStore * store = parent->icCommunicator()->publisherStore(getLayerId());
////   store->initializeThreadBuffers(parent);
//#endif

   return PV_SUCCESS;
}

//#ifdef PV_USE_OPENCL
///**
// * Initialize OpenCL buffers.  This must be called after PVLayer data have
// * been allocated.
// */
//int LIF::initializeThreadBuffers(const char * kernel_name)
//{
//   int status = HyPerLayer::initializeThreadBuffers(kernel_name);
//
//   const size_t size    = getNumNeurons()  * sizeof(pvdata_t);
//
//   CLDevice * device = parent->getCLDevice();
//
//   // these buffers are shared between host and device
//   //
//
//   clParams = device->createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(lParams), &lParams);
////   clParams->copyToDevice(&evUpdate);
////   status |= clWaitForEvents(1, &evUpdate);
////   clReleaseEvent(evUpdate);
//
//   clRand = device->createBuffer(CL_MEM_COPY_HOST_PTR, getNumNeurons()*sizeof(taus_uint4), rand_state);
//   clVth  = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, Vth);
//   clG_E  = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, G_E);
//   clG_I  = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, G_I);
//   clG_IB = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, G_IB);
//
//   return status;
//}
//
//int LIF::initializeThreadKernels(const char * kernel_name)
//{
//   char kernelPath[PV_PATH_MAX+128];
//   char kernelFlags[PV_PATH_MAX+128];
//
//   int status = CL_SUCCESS;
//   CLDevice * device = parent->getCLDevice();
//
//   const char * pvRelPath = "../PetaVision";
//   sprintf(kernelPath, "%s/%s/src/kernels/%s.cl", parent->getSrcPath(), pvRelPath, kernel_name);
//   sprintf(kernelFlags, "-D PV_USE_OPENCL -D USE_CLRANDOM -cl-fast-relaxed-math -I %s/%s/src/kernels/", parent->getSrcPath(), pvRelPath);
//
//   // create kernels
//   //
//   krUpdate = device->createKernel(kernelPath, kernel_name, kernelFlags);
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
//   status |= krUpdate->setKernelArg(argid++, clRand);
//
//   status |= krUpdate->setKernelArg(argid++, clV);
//   status |= krUpdate->setKernelArg(argid++, clVth);
//   status |= krUpdate->setKernelArg(argid++, clG_E);
//   status |= krUpdate->setKernelArg(argid++, clG_I);
//   status |= krUpdate->setKernelArg(argid++, clG_IB);
////   for (int i = 0; i < getNumChannels(); i++) {
////      status |= krUpdate->setKernelArg(argid++, clGSyn[i]);
////   }
//   status |= krUpdate->setKernelArg(argid++, clGSyn);
//   status |= krUpdate->setKernelArg(argid++, clActivity);
//   numKernelArgs = argid;
//
//   return status;
//}
//#endif

// Set Parameters
//

int LIF::ioParamsFillGroup(enum ParamsIOFlag ioFlag)
{
   HyPerLayer::ioParamsFillGroup(ioFlag);

   // clayer->params = &lParams; // Moved to initialize, after HyPerLayer::initialize call, since clayer isn't initialized until after ioParams is called.

   ioParam_Vrest(ioFlag);
   ioParam_Vexc(ioFlag);
   ioParam_Vinh(ioFlag);
   ioParam_VinhB(ioFlag);
   ioParam_VthRest(ioFlag);
   ioParam_tau(ioFlag);
   ioParam_tauE(ioFlag);
   ioParam_tauI(ioFlag);
   ioParam_tauIB(ioFlag);
   ioParam_tauVth(ioFlag);
   ioParam_deltaVth(ioFlag);
   ioParam_deltaGIB(ioFlag);

   // NOTE: in LIFDefaultParams, noise ampE, ampI, ampIB were
   // ampE=0*NOISE_AMP*( 1.0/TAU_EXC )
   //       *(( TAU_INH * (V_REST-V_INH) + TAU_INHB * (V_REST-V_INHB) ) / (V_EXC-V_REST))
   // ampI=0*NOISE_AMP*1.0
   // ampIB=0*NOISE_AMP*1.0
   // 

   ioParam_noiseAmpE(ioFlag);
   ioParam_noiseAmpI(ioFlag);
   ioParam_noiseAmpIB(ioFlag);
   ioParam_noiseFreqE(ioFlag);
   ioParam_noiseFreqI(ioFlag);
   ioParam_noiseFreqIB(ioFlag);

   ioParam_method(ioFlag);
   return 0;
}
void LIF::ioParam_Vrest(enum ParamsIOFlag ioFlag) { parent->ioParamValue(ioFlag, name, "Vrest", &lParams.Vrest, (float) V_REST); }
void LIF::ioParam_Vexc(enum ParamsIOFlag ioFlag) { parent->ioParamValue(ioFlag, name, "Vexc", &lParams.Vexc, (float) V_EXC); }
void LIF::ioParam_Vinh(enum ParamsIOFlag ioFlag) { parent->ioParamValue(ioFlag, name, "Vinh", &lParams.Vinh, (float) V_INH); }
void LIF::ioParam_VinhB(enum ParamsIOFlag ioFlag) { parent->ioParamValue(ioFlag, name, "VinhB", &lParams.VinhB, (float) V_INHB); }
void LIF::ioParam_VthRest(enum ParamsIOFlag ioFlag) { parent->ioParamValue(ioFlag, name, "VthRest", &lParams.VthRest, (float) VTH_REST); }
void LIF::ioParam_tau(enum ParamsIOFlag ioFlag) { parent->ioParamValue(ioFlag, name, "tau", &lParams.tau, (float) TAU_VMEM); }
void LIF::ioParam_tauE(enum ParamsIOFlag ioFlag) { parent->ioParamValue(ioFlag, name, "tauE", &lParams.tauE, (float) TAU_EXC); }
void LIF::ioParam_tauI(enum ParamsIOFlag ioFlag) { parent->ioParamValue(ioFlag, name, "tauI", &lParams.tauI, (float) TAU_INH); }
void LIF::ioParam_tauIB(enum ParamsIOFlag ioFlag) { parent->ioParamValue(ioFlag, name, "tauIB", &lParams.tauIB, (float) TAU_INHB); }
void LIF::ioParam_tauVth(enum ParamsIOFlag ioFlag) { parent->ioParamValue(ioFlag, name, "tauVth", &lParams.tauVth, (float) TAU_VTH); }
void LIF::ioParam_deltaVth(enum ParamsIOFlag ioFlag) { parent->ioParamValue(ioFlag, name, "deltaVth", &lParams.deltaVth, (float) DELTA_VTH); }
void LIF::ioParam_deltaGIB(enum ParamsIOFlag ioFlag) { parent->ioParamValue(ioFlag, name, "deltaGIB", &lParams.deltaGIB, (float) DELTA_G_INHB); }
void LIF::ioParam_noiseAmpE(enum ParamsIOFlag ioFlag) { parent->ioParamValue(ioFlag, name, "noiseAmpE", &lParams.noiseAmpE, 0.0f); }
void LIF::ioParam_noiseAmpI(enum ParamsIOFlag ioFlag) { parent->ioParamValue(ioFlag, name, "noiseAmpI", &lParams.noiseAmpI, 0.0f); }
void LIF::ioParam_noiseAmpIB(enum ParamsIOFlag ioFlag) { parent->ioParamValue(ioFlag, name, "noiseAmpIB", &lParams.noiseAmpIB, 0.0f); }

void LIF::ioParam_noiseFreqE(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "noiseFreqE", &lParams.noiseFreqE, 250.0f);
   if (ioFlag==PARAMS_IO_READ) {
      float dt_sec = .001 * parent->getDeltaTime();// seconds
      if (dt_sec * lParams.noiseFreqE  > 1.0) lParams.noiseFreqE  = 1.0/dt_sec;
   }
}

void LIF::ioParam_noiseFreqI(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "noiseFreqI", &lParams.noiseFreqI, 250.0f);
   if (ioFlag==PARAMS_IO_READ) {
      float dt_sec = .001 * parent->getDeltaTime();// seconds
      if (dt_sec * lParams.noiseFreqI  > 1.0) lParams.noiseFreqI  = 1.0/dt_sec;
   }
}

void LIF::ioParam_noiseFreqIB(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "noiseFreqIB", &lParams.noiseFreqIB, 250.0f);
   if (ioFlag==PARAMS_IO_READ) {
      float dt_sec = .001 * parent->getDeltaTime();// seconds
      if (dt_sec * lParams.noiseFreqIB > 1.0) lParams.noiseFreqIB = 1.0/dt_sec;
   }
}

void LIF::ioParam_method(enum ParamsIOFlag ioFlag) {
   // Read the integration method: one of 'arma' (preferred), 'beginning' (deprecated), or 'original' (deprecated).
   const char * default_method = "arma";
   parent->ioParamString(ioFlag, name, "method", &methodString, default_method, true/*warnIfAbsent*/);
   if (ioFlag != PARAMS_IO_READ) return;

   assert(methodString);
   if (methodString[0] == '\0') {
      free(methodString);
      methodString = strdup(default_method);
      if (methodString==NULL) {
         fprintf(stderr, "%s \"%s\" error: unable to set method string: %s\n", getKeyword(), name, strerror(errno));
         exit(EXIT_FAILURE);
      }
   }
   method = methodString ? methodString[0] : 'a'; // Default is ARMA; 'beginning' and 'original' are deprecated.
   if (method != 'o' && method != 'b' && method != 'a') {
      if (getParent()->columnId()==0) {
         fprintf(stderr, "LIF::setLIFParams error.  Layer \"%s\" has method \"%s\".  Allowable values are \"arma\", \"beginning\" and \"original\".", name, methodString);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   if (method != 'a') {
      if (getParent()->columnId()==0) {
         fprintf(stderr, "Warning: LIF layer \"%s\" integration method \"%s\" is deprecated.  Method \"arma\" is preferred.\n", name, methodString);
      }
   }
}

int LIF::setActivity() {
   pvdata_t * activity = clayer->activity->data;
   memset(activity, 0, sizeof(pvdata_t) * clayer->numExtendedAllBatches);
   return 0;
}

int LIF::communicateInitInfo() {
   int status = HyPerLayer::communicateInitInfo();

   return status;
}

int LIF::allocateDataStructures() {
   int status = HyPerLayer::allocateDataStructures();

   // // a random state variable is needed for every neuron/clthread
   randState = new Random(parent, getLayerLoc(), false/*isExtended*/);
   if (randState == NULL) {
      fprintf(stderr, "LIF::initialize error.  Layer \"%s\" unable to create object of Random class.\n", getName());
      exit(EXIT_FAILURE);
   }

   int numNeurons = getNumNeuronsAllBatches();
   assert(Vth); // Allocated when HyPerLayer::allocateDataStructures() called allocateBuffers().
   for (size_t k = 0; k < numNeurons; k++){
      Vth[k] = lParams.VthRest; // lParams.VthRest is set in setLIFParams
   }
   return status;
}

int LIF::allocateBuffers() {
   int status = allocateConductances(numChannels);
   assert(status==PV_SUCCESS);
   Vth = (pvdata_t *) calloc((size_t) getNumNeuronsAllBatches(), sizeof(pvdata_t));
   if(Vth == NULL) {
      fprintf(stderr, "LIF layer \"%s\" rank %d process unable to allocate memory for Vth: %s\n",
              name, parent->columnId(), strerror(errno));
      exit(EXIT_FAILURE);
   }
   return HyPerLayer::allocateBuffers();
}

int LIF::allocateConductances(int num_channels) {
   assert(num_channels>=3); // Need exc, inh, and inhb at a minimum.
   const int numNeurons = getNumNeuronsAllBatches();
   G_E = (pvdata_t *) calloc((size_t) (getNumNeuronsAllBatches()*numChannels), sizeof(pvdata_t));
   if(G_E == NULL) {
      fprintf(stderr, "LIF layer \"%s\" rank %d process unable to allocate memory for %d conductances: %s\n",
              name, parent->columnId(), num_channels, strerror(errno));
      exit(EXIT_FAILURE);
   }

   G_I  = G_E + 1*numNeurons;
   G_IB = G_E + 2*numNeurons;
   return PV_SUCCESS;
}

int LIF::readStateFromCheckpoint(const char * cpDir, double * timeptr) {
   int status = HyPerLayer::readStateFromCheckpoint(cpDir, timeptr);
   status = readVthFromCheckpoint(cpDir, timeptr);
   status = readG_EFromCheckpoint(cpDir, timeptr);
   status = readG_IFromCheckpoint(cpDir, timeptr);
   status = readG_IBFromCheckpoint(cpDir, timeptr);
   status = readRandStateFromCheckpoint(cpDir, timeptr);

   return PV_SUCCESS;
}

int LIF::readVthFromCheckpoint(const char * cpDir, double * timeptr) {
   char * filename = parent->pathInCheckpoint(cpDir, getName(), "_Vth.pvp");
   int status = readBufferFile(filename, parent->icCommunicator(), timeptr, &Vth, 1, /*extended*/true, getLayerLoc());
   assert(status==PV_SUCCESS);
   free(filename);
   return status;
}

int LIF::readG_EFromCheckpoint(const char * cpDir, double * timeptr) {
   char * filename = parent->pathInCheckpoint(cpDir, getName(), "_G_E.pvp");
   int status = readBufferFile(filename, parent->icCommunicator(), timeptr, &G_E, 1, /*extended*/true, getLayerLoc());
   assert(status==PV_SUCCESS);
   free(filename);
   return status;
}

int LIF::readG_IFromCheckpoint(const char * cpDir, double * timeptr) {
   char * filename = parent->pathInCheckpoint(cpDir, getName(), "_G_I.pvp");
   int status = readBufferFile(filename, parent->icCommunicator(), timeptr, &G_I, 1, /*extended*/true, getLayerLoc());
   assert(status==PV_SUCCESS);
   free(filename);
   return status;
}

int LIF::readG_IBFromCheckpoint(const char * cpDir, double * timeptr) {
   char * filename = parent->pathInCheckpoint(cpDir, getName(), "_G_IB.pvp");
   int status = readBufferFile(filename, parent->icCommunicator(), timeptr, &G_IB, 1, /*extended*/true, getLayerLoc());
   assert(status==PV_SUCCESS);
   free(filename);
   return status;
}

int LIF::readRandStateFromCheckpoint(const char * cpDir, double * timeptr) {
   char * filename = parent->pathInCheckpoint(cpDir, getName(), "_rand_state.bin");
   int status = readRandState(filename, parent->icCommunicator(), randState->getRNG(0), getLayerLoc(), false /*extended*/); // TODO Make a method in Random class
   assert(status==PV_SUCCESS);
   free(filename);
   return status;
}

int LIF::checkpointWrite(const char * cpDir) {
   HyPerLayer::checkpointWrite(cpDir);
   InterColComm * icComm = parent->icCommunicator();
   double timed = (double) parent->simulationTime();
   int filenamesize = strlen(cpDir)+1+strlen(name)+16;
   // The +1 is for the slash between cpDir and name; the +16 needs to be large enough to hold the suffix (e.g. _rand_state.bin) plus the null terminator
   char * filename = (char *) malloc( filenamesize*sizeof(char) );
   assert(filename != NULL);
   int chars_needed;

   chars_needed = snprintf(filename, filenamesize, "%s/%s_Vth.pvp", cpDir, name);
   assert(chars_needed < filenamesize);
   writeBufferFile(filename, icComm, timed, &Vth, 1, /*extended*/false, getLayerLoc());

   chars_needed = snprintf(filename, filenamesize, "%s/%s_G_E.pvp", cpDir, name);
   assert(chars_needed < filenamesize);
   writeBufferFile(filename, icComm, timed, &G_E, 1, /*extended*/false, getLayerLoc());

   chars_needed = snprintf(filename, filenamesize, "%s/%s_G_I.pvp", cpDir, name);
   assert(chars_needed < filenamesize);
   writeBufferFile(filename, icComm, timed, &G_I, 1, /*extended*/false, getLayerLoc());

   chars_needed = snprintf(filename, filenamesize, "%s/%s_G_IB.pvp", cpDir, name);
   assert(chars_needed < filenamesize);
   writeBufferFile(filename, icComm, timed, &G_IB, 1, /*extended*/false, getLayerLoc());

   chars_needed = snprintf(filename, filenamesize, "%s/%s_rand_state.bin", cpDir, name);
   assert(chars_needed < filenamesize);
   writeRandState(filename, parent->icCommunicator(), randState->getRNG(0), getLayerLoc(), false /*extended*/, parent->getVerifyWrites()); // TODO Make a method in Random class

   free(filename);
   return PV_SUCCESS;
}

int LIF::updateStateOpenCL(double time, double dt)
{
   int status = 0;

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
////   status |= getChannelCLBuffer(CHANNEL_INHB)->copyFromDevice(1, &evUpdate, &evList[getEVGSynIB()]);
//   status |= clActivity->copyFromDevice(1, &evUpdate, &evList[getEVActivity()]);
//   numWait += 2;
//
//#if PV_CL_COPY_BUFFERS
//   status |= clGSynE    ->copyFromDevice(1, &evUpdate, &evList[EV_LIF_GSyn_E]);
//   status |= clGSynI    ->copyFromDevice(1, &evUpdate, &evList[EV_LIF_GSyn_I]);
//   status |= clGSynIB   ->copyFromDevice(1, &evUpdate, &evList[EV_LIF_GSyn_IB]);
//   status |= clActivity ->copyFromDevice(1, &evUpdate, &evList[EV_LIF_ACTIVITY]);
//   numWait += 4;
//#endif // PV_CL_COPY_BUFFERS
//#endif // PV_USE_OPENCL

   return status;
}

int LIF::waitOnPublish(InterColComm* comm)
{
   int status = HyPerLayer::waitOnPublish(comm);

   // copy activity to device
   //
//#ifdef PV_USE_OPENCL
//#if PV_CL_COPY_BUFFERS
//   publish_timer->start();
//
//   status |= clActivity->copyToDevice(&evList[EV_LIF_ACTIVITY]);
//   numWait += 1;
//
//   publish_timer->stop();
//#endif
//#endif

   return status;
}

int LIF::updateState(double time, double dt)
{
   int status = 0;
   update_timer->start();

//#ifdef PV_USE_OPENCL
//   if((gpuAccelerateFlag)&&(true)) {
//      updateStateOpenCL(time, dt);
//   }
//   else {
//#endif
      const int nx = clayer->loc.nx;
      const int ny = clayer->loc.ny;
      const int nf = clayer->loc.nf;
      const PVHalo * halo = &clayer->loc.halo;
      const int nbatch = clayer->loc.nbatch;

      pvdata_t * GSynHead   = GSyn[0];
      pvdata_t * activity = clayer->activity->data;

      switch (method) {
      case 'a':
         LIF_update_state_arma(nbatch, getNumNeurons(), time, dt, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up, &lParams, randState->getRNG(0), clayer->V, Vth,
               G_E, G_I, G_IB, GSynHead, activity);
         break;
      case 'b':
         LIF_update_state_beginning(nbatch, getNumNeurons(), time, dt, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up, &lParams, randState->getRNG(0), clayer->V, Vth,
               G_E, G_I, G_IB, GSynHead, activity);
         break;
      case 'o':
         LIF_update_state_original(nbatch, getNumNeurons(), time, dt, nx, ny, nf, halo->lt, halo->rt, halo->dn, halo->up, &lParams, randState->getRNG(0), clayer->V, Vth,
               G_E, G_I, G_IB, GSynHead, activity);
         break;
      default:
         assert(0);
         break;
      }
//#ifdef PV_USE_OPENCL
//   }
//#endif

   update_timer->stop();
   return status;
}

float LIF::getChannelTimeConst(enum ChannelType channel_type)
{
   clayer->params = &lParams;
   float channel_time_const = 0.0f;
   switch (channel_type) {
   case CHANNEL_EXC:
      channel_time_const = lParams.tauE;
      break;
   case CHANNEL_INH:
      channel_time_const = lParams.tauI;
      break;
   case CHANNEL_INHB:
      channel_time_const = lParams.tauIB;
      break;
   default:
      channel_time_const = 0.0f;
      break;
   }
   return channel_time_const;
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

BaseObject * createLIF(char const * name, HyPerCol * hc) {
   return hc ? new LIF(name, hc) : NULL;
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
#else
#  undef PV_USE_OPENCL
#  include "../kernels/LIF_update_state.cl"
#  define PV_USE_OPENCL
#endif

#ifdef __cplusplus
}
#endif
