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

void LIFGap_update_state(
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
    float * GSynExc,
    float * GSynInh,
    float * GSynInhB,
    float * activity,

    const float sum_gap,
    float * G_Gap,
    float * GSynGap

);


#ifdef __cplusplus
}
#endif


namespace PV {

LIFGap::LIFGap(const char* name, HyPerCol * hc)
  : LIF(name, hc, TypeLIFGap, MAX_CHANNELS+1)
{
   const char * kernel_name = "LIFGap_update_state";
   LIFGap::initialize(TypeLIFGap, kernel_name );
}

LIFGap::LIFGap(const char* name, HyPerCol * hc, PVLayerType type)
  : LIF(name, hc, type, MAX_CHANNELS+1)
{
   const char * kernel_name = "LIFGap_update_state";
   LIFGap::initialize(type, kernel_name);
}

LIFGap::~LIFGap()
{


#ifdef PV_USE_OPENCL
   delete clG_Gap;
   delete clGSynGap;
#endif

}


// Initialize this class
/*
 *
 */
int LIFGap::initialize(PVLayerType type, const char * kernel_name)
{
   int status = CL_SUCCESS;

   status = LIF::initialize(type, kernel_name);

   const size_t num_neurons = getNumNeurons();
   this->G_Gap  = G_E + 3*num_neurons;
   this->sumGap = 0.0f;

   return status;
}

#ifdef PV_USE_OPENCL
/**
 * Initialize OpenCL buffers.  This must be called after PVLayer data have
 * been allocated.
 */
int LIFGap::initializeThreadBuffers(char * kernel_name)
{
   int status = CL_SUCCESS;

   status = LIF::initializeThreadBuffers(kernel_name);

   const size_t size    = getNumNeurons()  * sizeof(pvdata_t);
   CLDevice * device = parent->getCLDevice();

   // these buffers are shared between host and device
   //

   // TODO - use constant memory
   clG_Gap = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, G_Gap);
   clGSynGap = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, getChannel(CHANNEL_GAP));

   return status;
}

int LIFGap::initializeThreadKernels(char * kernel_name)

{
   int status = CL_SUCCESS;

   status = LIF::initializeThreadKernels(kernel_name);

   int argid = getNumKernelArgs();

   status |= krUpdate->setKernelArg(argid++, sumGap);
   status |= krUpdate->setKernelArg(argid++, clG_Gap);
   status |= krUpdate->setKernelArg(argid++, clGSynGap);

   numKernelArgs = argid;


   return status;
}
#endif


int LIFGap::updateStateOpenCL(float time, float dt)
{
   int status = CL_SUCCESS;

#ifdef PV_USE_OPENCL
   status = LIF::updateStateOpenCL(time, dt);

   status |= clGSynGap->copyFromDevice(1, &evUpdate, &evList[EV_LIF_GSYN_GAP]);

   numWait += 1;
#endif

   return status;
}

int LIFGap::triggerReceive(InterColComm* comm)
{
   int status = CL_SUCCESS;
   status = LIF::triggerReceive(comm);

#ifdef PV_USE_OPENCL
   // copy data to device
   status |= clGSynGap->copyToDevice(&evList[EV_LIF_GSYN_GAP]);
   numWait += 1;
#endif

   return status;
}


int LIFGap::updateState(float time, float dt)
{
   int status = CL_SUCCESS;
   update_timer->start();

#ifndef PV_USE_OPENCL

   const int nx = clayer->loc.nx;
   const int ny = clayer->loc.ny;
   const int nf = clayer->loc.nf;
   const int nb = clayer->loc.nb;

   pvdata_t * GSynExc   = getChannel(CHANNEL_EXC);
   pvdata_t * GSynInh   = getChannel(CHANNEL_INH);
   pvdata_t * GSynInhB  = getChannel(CHANNEL_INHB);
   pvdata_t * GSynGap  = getChannel(CHANNEL_GAP);
   pvdata_t * activity = clayer->activity->data;

   LIFGap_update_state(time, dt, nx, ny, nf, nb, &lParams, rand_state, clayer->V, Vth, G_E,
         G_I, G_IB, GSynExc, GSynInh, GSynInhB, activity, sumGap, G_Gap, GSynGap);

#else

   status = updateStateOpenCL(time, dt);

#endif
   updateActiveIndices();
   update_timer->stop();
   return status;
}


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
   status = read(path, comm, &dtime, G_Gap, loc, PV_FLOAT_TYPE, extended, contiguous);
   assert(status == PV_SUCCESS);

   return status;

}

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
   status = write(path, comm, time, G_Gap, loc, PV_FLOAT_TYPE, extended, contiguous);


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
#endif

#ifdef __cplusplus
}
#endif
