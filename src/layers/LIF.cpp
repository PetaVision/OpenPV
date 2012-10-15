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
#include "../utils/cl_random.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void LIF_update_state_beginning(
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
//    float * GSynExc,
//    float * GSynInh,
//    float * GSynInhB,
    float * activity);

void LIF_update_state_original(
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
//    float * GSynExc,
//    float * GSynInh,
//    float * GSynInhB,
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
   initialize(name, hc, TypeLIFSimple, MAX_CHANNELS, "LIF_update_state");
#ifdef PV_USE_OPENCL
   if(gpuAccelerateFlag)
      initializeGPU();
#endif
}

LIF::LIF(const char * name, HyPerCol * hc, PVLayerType type) {
   initialize_base();
   initialize(name, hc, type, MAX_CHANNELS, "LIF_update_state");
#ifdef PV_USE_OPENCL
   if(gpuAccelerateFlag)
      initializeGPU();
#endif
}

LIF::LIF(const char * name, HyPerCol * hc, PVLayerType type, int num_channels) {
   initialize_base();
   initialize(name, hc, type, numChannels, "LIF_update_state");
#ifdef PV_USE_OPENCL
   if(gpuAccelerateFlag)
      initializeGPU();
#endif
}

LIF::~LIF() {
   if (numChannels > 0) {
      // conductances allocated contiguously so this frees all
      free(G_E);
   }
   free(Vth);
   free(rand_state);

#ifdef PV_USE_OPENCL
//hyperlayer is destroying these:
//   delete krUpdate;
//
//   free(evList);
//
//   delete clParams;
   if(gpuAccelerateFlag) {
      delete clRand;
      delete clVth;
      delete clG_E;
      delete clG_I;
      delete clG_IB;
   }
#endif

}

int LIF::initialize_base() {
   rand_state = NULL;
   Vth = NULL;
   G_E = NULL;
   G_I = NULL;
   G_IB = NULL;

#ifdef PV_USE_OPENCL
   clRand = NULL;
   clVth = NULL;
   clG_E = NULL;
   clG_I = NULL;
   clG_IB = NULL;
#endif // PV_USE_OPEN_CL

   return PV_SUCCESS;
}

// Initialize this class
/*
 *
 * setParams() is called first so that we read all control parameters
 * from the params file.
 *
 */
int LIF::initialize(const char * name, HyPerCol * hc, PVLayerType type, int num_channels, const char * kernel_name) {
   setParams(hc->parameters(), hc->getDeltaTime());
   HyPerLayer::initialize(name, hc, num_channels);
   clayer->layerType = type;
   const size_t numNeurons = getNumNeurons();

   for (size_t k = 0; k < numNeurons; k++){
      Vth[k] = lParams.VthRest; // lParams.VthRest is set in setParams
   }

   // random seed should be different for different layers
   unsigned int seed = (unsigned int) (parent->getRandomSeed() + getLayerId());

   // a random state variable is needed for every neuron/clthread
   rand_state = cl_random_init(numNeurons, seed);

   // initialize OpenCL parameters
   //
   //This stuff is commented out for now, but will be used later and added to
   //its own initializeGPU method
#ifdef PV_USE_OPENCL
   numEvents=NUM_LIF_EVENTS;
//   CLDevice * device = parent->getCLDevice();
//
//   // TODO - fix to use device and layer parameters
//   if (device->id() == 1) {
//      nxl = 1;  nyl = 1;
//   }
//   else {
//      nxl = 16; nyl = 8;
//   }
//
//   numWait = 0;
//   numEvents = getNumCLEvents(); //NUM_LIF_EVENTS;
//   evList = (cl_event *) malloc(numEvents*sizeof(cl_event));
//   assert(evList != NULL);
//
//   numKernelArgs = 0;
//   initializeThreadBuffers(kernel_name);
//   initializeThreadKernels(kernel_name);
//
//   DataStore * store = parent->icCommunicator()->publisherStore(getLayerId());
//   store->initializeThreadBuffers(parent);
#endif

   return PV_SUCCESS;
}

#ifdef PV_USE_OPENCL
/**
 * Initialize OpenCL buffers.  This must be called after PVLayer data have
 * been allocated.
 */
int LIF::initializeThreadBuffers(const char * kernel_name)
{
   int status = HyPerLayer::initializeThreadBuffers(kernel_name);

   const size_t size    = getNumNeurons()  * sizeof(pvdata_t);

   CLDevice * device = parent->getCLDevice();

   // these buffers are shared between host and device
   //

   // TODO - use constant memory --done.  did I do it correctly?
   clParams = device->createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(lParams), &lParams);
//   clParams->copyToDevice(&evUpdate);
//   status |= clWaitForEvents(1, &evUpdate);
//   clReleaseEvent(evUpdate);

   clRand = device->createBuffer(CL_MEM_COPY_HOST_PTR, getNumNeurons()*sizeof(uint4), rand_state);
   clVth  = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, Vth);
   clG_E  = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, G_E);
   clG_I  = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, G_I);
   clG_IB = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, G_IB);

   return status;
}

int LIF::initializeThreadKernels(const char * kernel_name)
{
   char kernelPath[PV_PATH_MAX+128];
   char kernelFlags[PV_PATH_MAX+128];

   int status = CL_SUCCESS;
   CLDevice * device = parent->getCLDevice();

   const char * pvRelPath = "../PetaVision";
   sprintf(kernelPath, "%s/%s/src/kernels/%s.cl", parent->getPath(), pvRelPath, kernel_name);
   sprintf(kernelFlags, "-D PV_USE_OPENCL -D USE_CLRANDOM -cl-fast-relaxed-math -I %s/%s/src/kernels/", parent->getPath(), pvRelPath);

   // create kernels
   //
   krUpdate = device->createKernel(kernelPath, kernel_name, kernelFlags);

   int argid = 0;

   status |= krUpdate->setKernelArg(argid++, getNumNeurons());
   status |= krUpdate->setKernelArg(argid++, parent->simulationTime());
   status |= krUpdate->setKernelArg(argid++, parent->getDeltaTime());

   status |= krUpdate->setKernelArg(argid++, clayer->loc.nx);
   status |= krUpdate->setKernelArg(argid++, clayer->loc.ny);
   status |= krUpdate->setKernelArg(argid++, clayer->loc.nf);
   status |= krUpdate->setKernelArg(argid++, clayer->loc.nb);

   status |= krUpdate->setKernelArg(argid++, clParams);
   status |= krUpdate->setKernelArg(argid++, clRand);

   status |= krUpdate->setKernelArg(argid++, clV);
   status |= krUpdate->setKernelArg(argid++, clVth);
   status |= krUpdate->setKernelArg(argid++, clG_E);
   status |= krUpdate->setKernelArg(argid++, clG_I);
   status |= krUpdate->setKernelArg(argid++, clG_IB);
//   for (int i = 0; i < getNumChannels(); i++) {
//      status |= krUpdate->setKernelArg(argid++, clGSyn[i]);
//   }
   status |= krUpdate->setKernelArg(argid++, clGSyn);
   status |= krUpdate->setKernelArg(argid++, clActivity);
   numKernelArgs = argid;

   return status;
}
#endif

// Set Parameters
//
int LIF::setParams(PVParams * p, float dt)
{
   float dt_sec = .001 * dt;// seconds

   clayer->params = &lParams;

   spikingFlag = (int) p->value(name, "spikingFlag", 1);
   assert(spikingFlag == 1);  // LIF is a spiking layer

   lParams.Vrest     = p->value(name, "Vrest", V_REST);
   lParams.Vexc      = p->value(name, "Vexc" , V_EXC);
   lParams.Vinh      = p->value(name, "Vinh" , V_INH);
   lParams.VinhB     = p->value(name, "VinhB", V_INHB);
   lParams.VthRest   = p->value(name, "VthRest",VTH_REST);
   lParams.tau       = p->value(name, "tau"  , TAU_VMEM);
   lParams.tauE      = p->value(name, "tauE" , TAU_EXC);
   lParams.tauI      = p->value(name, "tauI" , TAU_INH);
   lParams.tauIB     = p->value(name, "tauIB", TAU_INHB);
   lParams.tauVth    = p->value(name, "tauVth" , TAU_VTH);
   lParams.deltaVth  = p->value(name, "deltaVth" , DELTA_VTH);

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

   const char * methodstring = p->stringValue(name, "method", true);
   method = methodstring ? methodstring[0] : 'o'; // Default is original but could change to 'beginning' if 'original' is deprecated.
   if (method != 'o' && method != 'b') {
      if (getParent()->columnId()==0) {
         fprintf(stderr, "LIFGap::initialize error.  Layer \"%s\" has method \"%s\".  Allowable values are \"beginning\" and \"original\".", name, methodstring);
      }
      abort();
   }
   return 0;
}

int LIF::setActivity() {
   pvdata_t * activity = clayer->activity->data;
   memset(activity, 0, sizeof(pvdata_t) * clayer->numExtended);
   return 0;
}

int LIF::allocateBuffers() {
   const size_t numNeurons = getNumNeurons();
   G_E = G_I = G_IB = NULL;

   if (numChannels > 0) {
      G_E = (pvdata_t *) calloc(numNeurons*numChannels, sizeof(pvdata_t));
      assert(G_E != NULL);

      G_I  = G_E + 1*numNeurons;
      G_IB = G_E + 2*numNeurons;
   }
   Vth = (pvdata_t *) calloc(numNeurons, sizeof(pvdata_t));
   assert(Vth != NULL);
   return HyPerLayer::allocateBuffers();
}

int LIF::checkpointRead(const char * cpDir, float * timef) {
   HyPerLayer::checkpointRead(cpDir, timef);
   InterColComm * icComm = parent->icCommunicator();
   double timed;
   int filenamesize = strlen(name) + strlen(cpDir) + 13;
   // The +12 needs to be large enough to hold the suffix (e.g. _G_IB.pvp) plus the null terminator
   char * filename = (char *) malloc( filenamesize*sizeof(char) );
   assert(filename != NULL);

   int chars_needed = snprintf(filename, filenamesize, "%s/%s_Vth.pvp", cpDir, name);
   assert(chars_needed < filenamesize);
   readBufferFile(filename, icComm, &timed, Vth, 1, /*extended*/false, /*contiguous*/false, getLayerLoc());
   if( (float) timed != *timef && parent->icCommunicator()->commRank() == 0 ) {
      fprintf(stderr, "Warning: %s and %s_A.pvp have different timestamps: %f versus %f\n", filename, name, (float) timed, *timef);
   }

   chars_needed = snprintf(filename, filenamesize, "%s/%s_G_E.pvp", cpDir, name);
   assert(chars_needed < filenamesize);
   readBufferFile(filename, icComm, &timed, G_E, 1, /*extended*/false, /*contiguous*/false, getLayerLoc());
   if( (float) timed != *timef && parent->icCommunicator()->commRank() == 0 ) {
      fprintf(stderr, "Warning: %s and %s_A.pvp have different timestamps: %f versus %f\n", filename, name, (float) timed, *timef);
   }

   chars_needed = snprintf(filename, filenamesize, "%s/%s_G_I.pvp", cpDir, name);
   assert(chars_needed < filenamesize);
   readBufferFile(filename, icComm, &timed, G_I, 1, /*extended*/false, /*contiguous*/false, getLayerLoc());
   if( (float) timed != *timef && parent->icCommunicator()->commRank() == 0 ) {
      fprintf(stderr, "Warning: %s and %s_A.pvp have different timestamps: %f versus %f\n", filename, name, (float) timed, *timef);
   }

   chars_needed = snprintf(filename, filenamesize, "%s/%s_G_IB.pvp", cpDir, name);
   assert(chars_needed < filenamesize);
   readBufferFile(filename, icComm, &timed, G_IB, 1, /*extended*/false, /*contiguous*/false, getLayerLoc());
   if( (float) timed != *timef && parent->icCommunicator()->commRank() == 0 ) {
      fprintf(stderr, "Warning: %s and %s_A.pvp have different timestamps: %f versus %f\n", filename, name, (float) timed, *timef);
   }

   free(filename);
   return PV_SUCCESS;
}

int LIF::checkpointWrite(const char * cpDir) {
   HyPerLayer::checkpointWrite(cpDir);
   InterColComm * icComm = parent->icCommunicator();
   double timed = (double) parent->simulationTime();
   int filenamesize = strlen(cpDir)+1+strlen(name)+12;
   // The +1 is for the slash between cpDir and name; the +12 needs to be large enough to hold the suffix (e.g. _G_Gap.pvp) plus the null terminator
   char * filename = (char *) malloc( filenamesize*sizeof(char) );
   assert(filename != NULL);
   int chars_needed = snprintf(filename, filenamesize, "%s/%s_Vth.pvp", cpDir, name);
   assert(chars_needed < filenamesize);
   writeBufferFile(filename, icComm, timed, Vth, 1, /*extended*/false, /*contiguous*/false, getLayerLoc()); // TODO contiguous=true
   chars_needed = snprintf(filename, filenamesize, "%s/%s_G_E.pvp", cpDir, name);
   assert(chars_needed < filenamesize);
   writeBufferFile(filename, icComm, timed, G_E, 1, /*extended*/false, /*contiguous*/false, getLayerLoc()); // TODO contiguous=true
   chars_needed = snprintf(filename, filenamesize, "%s/%s_G_I.pvp", cpDir, name);
   assert(chars_needed < filenamesize);
   writeBufferFile(filename, icComm, timed, G_I, 1, /*extended*/false, /*contiguous*/false, getLayerLoc()); // TODO contiguous=true
   chars_needed = snprintf(filename, filenamesize, "%s/%s_G_IB.pvp", cpDir, name);
   assert(chars_needed < filenamesize);
   writeBufferFile(filename, icComm, timed, G_IB, 1, /*extended*/false, /*contiguous*/false, getLayerLoc()); // TODO contiguous=true
   free(filename);
   return PV_SUCCESS;
}

int LIF::updateStateOpenCL(float time, float dt)
{
   int status = CL_SUCCESS;

#ifdef PV_USE_OPENCL
   // wait for memory to be copied to device
   if (numWait > 0) {
       status |= clWaitForEvents(numWait, evList);
   }
   for (int i = 0; i < numWait; i++) {
      clReleaseEvent(evList[i]);
   }
   numWait = 0;

   status |= krUpdate->setKernelArg(1, time);
   status |= krUpdate->setKernelArg(2, dt);
   status |= krUpdate->run(getNumNeurons(), nxl*nyl, 0, NULL, &evUpdate);
   krUpdate->finish();

   status |= getChannelCLBuffer()->copyFromDevice(1, &evUpdate, &evList[getEVGSyn()]);
//   status |= getChannelCLBuffer(CHANNEL_EXC)->copyFromDevice(1, &evUpdate, &evList[getEVGSynE()]);
//   status |= getChannelCLBuffer(CHANNEL_INH)->copyFromDevice(1, &evUpdate, &evList[getEVGSynI()]);
//   status |= getChannelCLBuffer(CHANNEL_INHB)->copyFromDevice(1, &evUpdate, &evList[getEVGSynIB()]);
   status |= clActivity->copyFromDevice(1, &evUpdate, &evList[getEVActivity()]);
   numWait += 2;

#if PV_CL_COPY_BUFFERS
   status |= clGSynE    ->copyFromDevice(1, &evUpdate, &evList[EV_LIF_GSyn_E]);
   status |= clGSynI    ->copyFromDevice(1, &evUpdate, &evList[EV_LIF_GSyn_I]);
   status |= clGSynIB   ->copyFromDevice(1, &evUpdate, &evList[EV_LIF_GSyn_IB]);
   status |= clActivity ->copyFromDevice(1, &evUpdate, &evList[EV_LIF_ACTIVITY]);
   numWait += 4;
#endif // PV_CL_COPY_BUFFERS
#endif // PV_USE_OPENCL

   return status;
}

int LIF::triggerReceive(InterColComm* comm)
{
   int status = HyPerLayer::triggerReceive(comm);

   // copy data to device
   //
#ifdef PV_USE_OPENCL
//   if(gpuAccelerateFlag) {
//      status |= getChannelCLBuffer(CHANNEL_INHB)->copyToDevice(&evList[getEVGSynIB()]);
//      //status |= getChannelCLBuffer(CHANNEL_INHB)->copyToDevice(&evList[getEVGSynIB()]);
//      numWait += 1;
//   }
#if PV_CL_COPY_BUFFERS
   status |= clGSynE->copyToDevice(&evList[EV_LIF_GSYN_E]);
   status |= clGSynI->copyToDevice(&evList[EV_LIF_GSYN_I]);
   status |= clGSynI->copyToDevice(&evList[EV_LIF_GSYN_IB]);
   numWait += 3;
#endif
#endif

   return status;
}

int LIF::waitOnPublish(InterColComm* comm)
{
   int status = HyPerLayer::waitOnPublish(comm);

   // copy activity to device
   //
#ifdef PV_USE_OPENCL
#if PV_CL_COPY_BUFFERS
   status |= clActivity->copyToDevice(&evList[EV_LIF_ACTIVITY]);
   numWait += 1;
#endif
#endif

   return status;
}

int LIF::updateState(float time, float dt)
{
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

#ifdef OBSOLETE // Marked obsolete July 13, 2012.  Restarting from last now handled by a call to checkpointRead from within HyPerLayer::initializeState
int LIF::readState(float * time)
{
   double dtime;
   char path[PV_PATH_MAX];
   bool contiguous = false;
   bool extended   = false;

   int status = HyPerLayer::readState(time);

   PVLayerLoc * loc = & clayer->loc;
   Communicator * comm = parent->icCommunicator();

   getOutputFilename(path, "Vth", "_last");
   status = read_pvdata(path, comm, &dtime, Vth, loc, PV_FLOAT_TYPE, extended, contiguous);
   assert(status == PV_SUCCESS);

   getOutputFilename(path, "G_E", "_last");
   status = read_pvdata(path, comm, &dtime, G_E, loc, PV_FLOAT_TYPE, extended, contiguous);
   assert(status == PV_SUCCESS);

   getOutputFilename(path, "G_I", "_last");
   status = read_pvdata(path, comm, &dtime, G_I, loc, PV_FLOAT_TYPE, extended, contiguous);
   assert(status == PV_SUCCESS);

   getOutputFilename(path, "G_IB", "_last");
   status = read_pvdata(path, comm, &dtime, G_IB, loc, PV_FLOAT_TYPE, extended, contiguous);
   assert(status == PV_SUCCESS);

   *time = (float) dtime;
   return status;

}
#endif // OBSOLETE

#ifdef OBSOLETE // Marked obsolete Jul 13, 2012.  Dumping the state is now done by CheckpointWrite.
int LIF::writeState(float time, bool last)
{
   char path[PV_PATH_MAX];
   bool contiguous = false;
   bool extended   = false;

   const char * last_str = (last) ? "_last" : "";

   int status = HyPerLayer::writeState(time, last);

   PVLayerLoc * loc = & clayer->loc;
   Communicator * comm = parent->icCommunicator();

   getOutputFilename(path, "Vth", last_str);
   status = write_pvdata(path, comm, time, Vth, loc, PV_FLOAT_TYPE, extended, contiguous);

   getOutputFilename(path, "G_E", last_str);
   status = write_pvdata(path, comm, time, G_E, loc, PV_FLOAT_TYPE, extended, contiguous);

   getOutputFilename(path, "G_I", last_str);
   status = write_pvdata(path, comm, time, G_I, loc, PV_FLOAT_TYPE, extended, contiguous);

   getOutputFilename(path, "G_IB", last_str);
   status = write_pvdata(path, comm, time, G_IB, loc, PV_FLOAT_TYPE, extended, contiguous);

#ifdef DEBUG_OUTPUT
   // print activity at center of image

   int sx = clayer->loc.nf;
   int sy = sx*clayer->loc.nx;
   pvdata_t * a = clayer->activity->data;

   int n = (int) (sy*(clayer->loc.ny/2 - 1) + sx*(clayer->loc.nx/2));
   for (int f = 0; f < clayer->loc.nf; f++) {
      printf("f = %d, a[%d] = %f\n", f, n, a[n]);
      n += 1;
   }
   printf("\n");

   n = (int) (sy*(clayer->loc.ny/2 - 1) + sx*(clayer->loc.nx/2));
   n -= 8;
   for (int f = 0; f < clayer->loc.nf; f++) {
      printf("f = %d, a[%d] = %f\n", f, n, a[n]);
      n += 1;
   }
#endif

   return 0;
}
#endif // OBSOLETE

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

//#ifndef PV_USE_OPENCL
//#  include "../kernels/LIF_update_state.cl"
//#endif
//#include "../kernels/LIF_update_state.c"
#ifndef PV_USE_OPENCL
#  include "../kernels/LIF_update_state.cl"
#else
#  undef PV_USE_OPENCL
#  undef USE_CLRANDOM
#  include "../kernels/LIF_update_state.cl"
#  define PV_USE_OPENCL
#endif

#ifdef __cplusplus
}
#endif
