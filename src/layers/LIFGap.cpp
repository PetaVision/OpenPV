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
//    float * GSynExc,
//    float * GSynInh,
//    float * GSynInhB,
//    float * GSynGap,
    float * activity,

    const float sum_gap,
    float * G_Gap
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
//    float * GSynExc,
//    float * GSynInh,
//    float * GSynInhB,
//    float * GSynGap,
    float * activity,

    const float sum_gap,
    float * G_Gap
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
   initialize(name, hc, TypeLIFGap, MAX_CHANNELS+1, "LIFGap_update_state");
#ifdef PV_USE_OPENCL
   if(gpuAccelerateFlag)
      initializeGPU();
#endif
}

LIFGap::LIFGap(const char * name, HyPerCol * hc, PVLayerType type) {
   initialize_base();
   initialize(name, hc, type, MAX_CHANNELS+1, "LIFGap_update_state");
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
   G_Gap = NULL;

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
int LIFGap::initialize(const char * name, HyPerCol * hc, PVLayerType type, int num_channels, const char * kernel_name) {
   int status = LIF::initialize(name, hc, type, num_channels, kernel_name);

   // Initialization of method moved to LIF::setParams
   // PVParams * params = hc->parameters();
   // const char * methodstring = params->stringValue(name, "method", true);
   // method = methodstring ? methodstring[0] : 'o';
   // if (method != 'o' && method != 'b') {
   //    if (hc->icCommunicator()->commRank()==0) {
   //       fprintf(stderr, "LIFGap::initialize error.  Layer \"%s\" has method \"%s\".  Allowable values are \"beginning\" and \"original\".", name, methodstring);
   //    }
   //    abort();
   // }

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

   // TODO - use constant memory ????  put what in constant memory???
   clG_Gap = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, G_Gap);
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

int LIFGap::allocateBuffers() {
   int status = LIF::allocateBuffers();
   if( status == PV_SUCCESS ) {
      const size_t num_neurons = getNumNeurons();
      this->G_Gap  = G_E + 3*num_neurons;
      this->sumGap = 0.0f;
   }
   return status;
}

int LIFGap::checkpointRead(const char * cpDir, float * timef) {
   LIF::checkpointRead(cpDir, timef);
   InterColComm * icComm = parent->icCommunicator();
   double timed;
   int filenamesize = strlen(cpDir)+1+strlen(name)+12;
   // The +1 is for the slash between cpDir and name; the +12 needs to be large enough to hold the suffix (e.g. _G_Gap.pvp) plus the null terminator
   char * filename = (char *) malloc( filenamesize*sizeof(char) );
   assert(filename != NULL);

   int chars_needed = snprintf(filename, filenamesize, "%s/%s_G_Gap.pvp", cpDir, name);
   assert(chars_needed < filenamesize);
   readBufferFile(filename, icComm, &timed, G_Gap, 1, /*extended*/false, /*contiguous*/false);
   if( (float) timed != *timef && parent->icCommunicator()->commRank() == 0 ) {
      fprintf(stderr, "Warning: %s and %s_A.pvp have different timestamps: %f versus %f\n", filename, name, (float) timed, *timef);
   }

   free(filename);
   return PV_SUCCESS;
}

int LIFGap::checkpointWrite(const char * cpDir) {
   LIF::checkpointWrite(cpDir);
   InterColComm * icComm = parent->icCommunicator();
   double timed = (double) parent->simulationTime();
   int filenamesize = strlen(cpDir)+1+strlen(name)+12;
   // The +1 is for the slash between cpDir and name; the +12 needs to be large enough to hold the suffix (e.g. _G_Gap.pvp) plus the null terminator
   char * filename = (char *) malloc( filenamesize*sizeof(char) );
   assert(filename != NULL);
   sprintf(filename, "%s/%s_G_Gap.pvp", cpDir, name);
   writeBufferFile(filename, icComm, timed, G_Gap, 1, /*extended*/false, /*contiguous*/false); // TODO contiguous=true
   free(filename);
   return PV_SUCCESS;
}

int LIFGap::updateStateOpenCL(float time, float dt)
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

int LIFGap::triggerReceive(InterColComm* comm)
{
   int status = CL_SUCCESS;
   status = LIF::triggerReceive(comm);

#ifdef PV_USE_OPENCL
   // copy data to device
//   if(gpuAccelerateFlag) {
//      //do we need to copy gap back and forth? what should the event be?
//      //status |= clG_Gap->copyToDevice(1, &evUpdate, &evList[EV_LIF_GSYN_GAP]);
//      status |= getChannelCLBuffer(CHANNEL_GAP)->copyToDevice(&evList[getEVGSynGap()]);
//      numWait += 1;
//   }
#if PV_CL_COPY_BUFFERS
   status |= clGSynGap->copyToDevice(&evList[EV_LIF_GSYN_GAP]);
   numWait += 1;
#endif
#endif

   return status;
}


int LIFGap::updateState(float time, float dt)
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
//   pvdata_t * GSynExc   = getChannel(CHANNEL_EXC);
//   pvdata_t * GSynInh   = getChannel(CHANNEL_INH);
//   pvdata_t * GSynInhB  = getChannel(CHANNEL_INHB);
//   pvdata_t * GSynGap  = getChannel(CHANNEL_GAP);
   pvdata_t * activity = clayer->activity->data;

   switch (method) {
   case 'b':
      LIFGap_update_state_beginning(getNumNeurons(), time, dt, nx, ny, nf, nb, &lParams, rand_state, clayer->V, Vth, G_E,
            G_I, G_IB, GSynHead, activity, sumGap, G_Gap);
   break;
   case 'o':
      LIFGap_update_state_original(getNumNeurons(), time, dt, nx, ny, nf, nb, &lParams, rand_state, clayer->V, Vth, G_E,
            G_I, G_IB, GSynHead, activity, sumGap, G_Gap);
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
int LIFGap::readState(float * time)
{
   double dtime;
   char path[PV_PATH_MAX];
   bool contiguous = false;
   bool extended   = false;

   int status = LIF::readState(time);

   PVLayerLoc * loc = & clayer->loc;
   Communicator * comm = parent->icCommunicator();

   getOutputFilename(path, "G_Gap", "_last");
   status = read_pvdata(path, comm, &dtime, G_Gap, loc, PV_FLOAT_TYPE, extended, contiguous);
   assert(status == PV_SUCCESS);

   return status;

}
#endif // OBSOLETE

#ifdef OBSOLETE // Marked obsolete Jul 13, 2012.  Dumping the state is now done by CheckpointWrite.
int LIFGap::writeState(float time, bool last)
{
   char path[PV_PATH_MAX];
   bool contiguous = false;
   bool extended   = false;

   const char * last_str = (last) ? "_last" : "";

   int status = LIF::writeState(time, last);

   PVLayerLoc * loc = & clayer->loc;
   Communicator * comm = parent->icCommunicator();

   getOutputFilename(path, "G_Gap", last_str);
   status = write_pvdata(path, comm, time, G_Gap, loc, PV_FLOAT_TYPE, extended, contiguous);


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
#  undef USE_CLRANDOM
#  include "../kernels/LIFGap_update_state.cl"
#  define PV_USE_OPENCL
#endif

#ifdef __cplusplus
}
#endif
