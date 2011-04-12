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
#include "../connections/PVConnection.h"
#include "../io/fileio.hpp"
#include "../utils/cl_random.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// only for my dumb debugging
#include <iostream>
using namespace std;

#ifdef __cplusplus
extern "C" {
#endif

void LIF_update_state(
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
    float * phiExc,
    float * phiInh,
    float * phiInhB,
    float * R,
    float * activity);

void LIF_update_state_localWmax(
    const float time,
    const float dt,

    const int nx,
    const int ny,
    const int nf,
    const int nb,

    const float gammaW,
    const float alphaW,
    const float averageR,

    LIF_params * params,
    uint4 * rnd,

    float * V,
    float * Vth,
    float * G_E,
    float * G_I,
    float * G_IB,
    float * phiExc,
    float * phiInh,
    float * phiInhB,
    float * R,
    float * Wmax,
    float * activity);

#ifdef __cplusplus
}
#endif

namespace PV
{

#ifdef OBSOLETE
LIFParams LIFDefaultParams =
{
    V_REST, V_EXC, V_INH, V_INHB,            // V (mV)
    TAU_VMEM, TAU_EXC, TAU_INH, TAU_INHB,
    VTH_REST,  TAU_VTH, DELTA_VTH,	     // tau (ms)
    250, 0*NOISE_AMP*( 1.0/TAU_EXC ) * ( ( TAU_INH * (V_REST-V_INH) + TAU_INHB * (V_REST-V_INHB) ) / (V_EXC-V_REST) ),
    250, 0*NOISE_AMP*1.0,
    250, 0*NOISE_AMP*1.0                       // noise (G)
};
#endif

LIF::LIF(const char* name, HyPerCol * hc)
  : HyPerLayer(name, hc, MAX_CHANNELS)
{
   initialize(TypeLIFSimple);
}

LIF::LIF(const char* name, HyPerCol * hc, PVLayerType type)
  : HyPerLayer(name, hc, MAX_CHANNELS)
{
   initialize(type);
}

LIF::~LIF()
{
   if (numChannels > 0) {
      // conductances allocated contiguously so this frees all
      free(G_E);
   }
   free(Vth);
   free(rand_state);
   free(R);

   if(localWmaxFlag){
      free(Wmax);
   }
#ifdef PV_USE_OPENCL
   free(evList);

   delete clParams;
   delete clRand;
   delete clV;
   delete clVth;
   delete clG_E;
   delete clG_I;
   delete clG_IB;
   delete clPhiE;
   delete clPhiI;
   delete clPhiIB;
   delete clR;
   delete clActivity;
   delete clPrevTime;
#endif

}


// Initialize this class
/*
 *
 * setParams() is called first so that we read all control parameters
 * (rateFlag, tauRate, writeRate, etc) from the params file.
 * Wmax is an extended variable so that it matches the decrement variable M in
 * the HyPerConn class (M is a post layer extended variable)
 * R (rate) is a restricted variable
 */
int LIF::initialize(PVLayerType type)
{
   float time = 0.0f;
   int status = CL_SUCCESS;

   const size_t numNeurons = getNumNeurons();

   setParams(parent->parameters());
   clayer->layerType = type;

   G_E = G_I = G_IB = NULL;

   if (numChannels > 0) {
      G_E = (pvdata_t *) calloc(numNeurons*numChannels, sizeof(pvdata_t));
      assert(G_E != NULL);

      G_I  = G_E + 1*numNeurons;
      G_IB = G_E + 2*numNeurons;
   }

   // a random state variable is needed for every neuron/clthread
   rand_state = cl_random_init(numNeurons);

   // initialize layer data
   //
   Vth = (pvdata_t *) calloc(numNeurons, sizeof(pvdata_t));
   assert(Vth != NULL);
   for (size_t k = 0; k < numNeurons; k++){
      Vth[k] = VTH_REST;
   }

   // allocate memory for R
   //
   R = (pvdata_t *) calloc(numNeurons, sizeof(pvdata_t) );
   assert(R != NULL);

   // allocate memory for wMax
   if(localWmaxFlag){
      const size_t numExtended = getNumExtended();
      Wmax = (pvdata_t *) calloc(numExtended, sizeof(pvdata_t) );
      assert(Wmax != NULL);
      cout << "Wmax pointer in LIF: " << Wmax << endl;
      for (size_t k = 0; k < numExtended; k++){
         Wmax[k] = wMax;
      }
   }else{
      Wmax = NULL;
   }

   parent->addLayer(this);

   if (parent->parameters()->value(name, "restart", 0) != 0) {
      readState(&time);
   }

   // initialize OpenCL parameters
   //
#ifdef PV_USE_OPENCL
   CLDevice * device = parent->getCLDevice();

   // TODO - fix to use device and layer parameters
   if (device->id() == 1) {
      nxl = 1;  nyl = 1;
   }
   else {
      nxl = 16; nyl = 8;
   }

   numWait = 0;
   numEvents = NUM_LIF_EVENTS;
   evList = (cl_event *) malloc(numEvents*sizeof(cl_event));
   assert(evList != NULL);

   initializeThreadBuffers();
   initializeThreadKernels();
#endif

   return status;
}

#ifdef PV_USE_OPENCL
/**
 * Initialize OpenCL buffers.  This must be called after PVLayer data have
 * been allocated.
 */
int LIF::initializeThreadBuffers()
{
   int status = CL_SUCCESS;

   const size_t size    = getNumNeurons()  * sizeof(pvdata_t);
   const size_t size_ex = getNumExtended() * sizeof(pvdata_t);

   CLDevice * device = parent->getCLDevice();

   // these buffers are shared between host and device
   //

   // TODO - use constant memory
   clParams = device->createBuffer(CL_MEM_COPY_HOST_PTR, sizeof(lParams), &lParams);
   clRand   = device->createBuffer(CL_MEM_COPY_HOST_PTR, getNumNeurons()*sizeof(uint4), rand_state);

   clV    = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, clayer->V);
   clVth  = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, Vth);
   clG_E  = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, G_E);
   clG_I  = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, G_I);
   clG_IB = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, G_IB);

   clPhiE  = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, getChannel(CHANNEL_EXC));
   clPhiI  = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, getChannel(CHANNEL_INH));
   clPhiIB = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, getChannel(CHANNEL_INHB));

   clR = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, R);

   clActivity = device->createBuffer(CL_MEM_COPY_HOST_PTR, size_ex, clayer->activity->data);
   clPrevTime = device->createBuffer(CL_MEM_COPY_HOST_PTR, size_ex, clayer->prevActivity);

   return status;
}

int LIF::initializeThreadKernels()
{
   char kernelPath[PV_PATH_MAX+128];
   char kernelFlags[PV_PATH_MAX+128];

   int status = CL_SUCCESS;
   CLDevice * device = parent->getCLDevice();

   sprintf(kernelPath, "%s/src/kernels/LIF_update_state.cl", parent->getPath());
   sprintf(kernelFlags, "-D PV_USE_OPENCL -cl-fast-relaxed-math -I %s/src/kernels/", parent->getPath());

   // create kernels
   //
   krUpdate = device->createKernel(kernelPath, "LIF_update_state", kernelFlags);

   int argid = 0;

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
   status |= krUpdate->setKernelArg(argid++, clPhiE);
   status |= krUpdate->setKernelArg(argid++, clPhiI);
   status |= krUpdate->setKernelArg(argid++, clPhiIB);
   status |= krUpdate->setKernelArg(argid++, clR);
   status |= krUpdate->setKernelArg(argid++, clActivity);

   return status;
}
#endif

// Set Parameters
//
int LIF::setParams(PVParams * p)
{
   float dt_sec = .001 * parent->getDeltaTime();  // seconds

   clayer->params = &lParams;

   spikingFlag = (int) p->value(name, "spikingFlag", 1);
   assert(spikingFlag == 1);  // LIF is a spiking layer

   lParams.Vrest = p->value(name, "Vrest", V_REST);
   lParams.Vexc  = p->value(name, "Vexc" , V_EXC);
   lParams.Vinh  = p->value(name, "Vinh" , V_INH);
   lParams.VinhB = p->value(name, "VinhB", V_INHB);

   lParams.tau   = p->value(name, "tau"  , TAU_VMEM);
   lParams.tauE  = p->value(name, "tauE" , TAU_EXC);
   lParams.tauI  = p->value(name, "tauI" , TAU_INH);
   lParams.tauIB = p->value(name, "tauIB", TAU_INHB);

   lParams.tauRate  = p->value(name, "tauRate",  TAU_RATE);
   lParams.VthRest  = p->value(name, "VthRest" , VTH_REST);
   lParams.tauVth   = p->value(name, "tauVth"  , TAU_VTH);
   lParams.deltaVth = p->value(name, "deltaVth", DELTA_VTH);

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
   
   // set wMax parameters
   wMax = p->value(name, "wMax", 0.75);
   wMin = p->value(name, "wMin", 0.0);


   // set params for rate dependent Wmax
   localWmaxFlag = false;
   localWmaxFlag = (bool) p->value(name, "localWmaxFlag", (float) localWmaxFlag);
   tauWmax     = p->value(name,"tauWmax",TAU_WMAX); // in ms
   alphaW     = p->value(name,"alphaW",0.01);
   averageR   = p->value(name,"averageR",10.0);

   return 0;
}

int LIF::updateStateOpenCL(float time, float dt)
{
   int status = CL_SUCCESS;

#ifdef PV_USE_OPENCL
   // wait for memory to be copied to device
   status |= clWaitForEvents(numWait, evList);
   for (int i = 0; i < numWait; i++) {
      clReleaseEvent(evList[i]);
   }
   numWait = 0;

   status |= krUpdate->setKernelArg(0, time);
   status |= krUpdate->setKernelArg(1, dt);
   status |= krUpdate->run(getNumNeurons(), nxl*nyl, 0, NULL, &evUpdate);

   status |= clPhiE    ->copyFromDevice(1, &evUpdate, &evList[EV_LIF_PHI_E]);
   status |= clPhiI    ->copyFromDevice(1, &evUpdate, &evList[EV_LIF_PHI_I]);
   status |= clPhiIB   ->copyFromDevice(1, &evUpdate, &evList[EV_LIF_PHI_IB]);
   status |= clActivity->copyFromDevice(1, &evUpdate, &evList[EV_LIF_ACTIVITY]);

   numWait += 4;
#endif

   return status;
}

int LIF::triggerReceive(InterColComm* comm)
{
   int status = HyPerLayer::triggerReceive(comm);

   // copy data to device
   //
#ifdef PV_USE_OPENCL
   status |= clPhiE->copyToDevice(&evList[EV_LIF_PHI_E]);
   status |= clPhiI->copyToDevice(&evList[EV_LIF_PHI_I]);
   status |= clPhiI->copyToDevice(&evList[EV_LIF_PHI_IB]);
   numWait += 3;
#endif

   return status;
}

int LIF::waitOnPublish(InterColComm* comm)
{
   int status = HyPerLayer::waitOnPublish(comm);

   // copy activity to device
   //
#ifdef PV_USE_OPENCL
   status |= clActivity->copyToDevice(&evList[EV_LIF_ACTIVITY]);
   numWait += 1;
#endif

   return status;
}

int LIF::updateState(float time, float dt)
{
   int status = 0;
   update_timer->start();

#ifndef PV_USE_OPENCL

   const int nx = clayer->loc.nx;
   const int ny = clayer->loc.ny;
   const int nf = clayer->loc.nf;
   const int nb = clayer->loc.nb;

   pvdata_t * phiExc   = getChannel(CHANNEL_EXC);
   pvdata_t * phiInh   = getChannel(CHANNEL_INH);
   pvdata_t * phiInhB  = getChannel(CHANNEL_INHB);
   pvdata_t * activity = clayer->activity->data;

   if(localWmaxFlag){
      //float tauRate  = parent->parameters()->value(getName(), "tauRate", TAU_RATE);
      LIF_update_state_localWmax(time, dt, nx, ny, nf, nb,
                          tauWmax,alphaW,averageR,
                          &lParams, rand_state,
                          clayer->V, Vth,
                          G_E, G_I, G_IB,
                          phiExc, phiInh, phiInhB, R, Wmax, activity);
   } else {
      LIF_update_state(time, dt, nx, ny, nf, nb,
                       &lParams, rand_state,
                       clayer->V, Vth,
                       G_E, G_I, G_IB,
                       phiExc, phiInh, phiInhB, R, activity);
   }
#else

   status = updateStateOpenCL(time, dt);

#endif

   update_timer->stop();
   return status;
}

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
   status = read(path, comm, &dtime, Vth, loc, PV_FLOAT_TYPE, extended, contiguous);
   assert(status == PV_SUCCESS);

   getOutputFilename(path, "G_E", "_last");
   status = read(path, comm, &dtime, G_E, loc, PV_FLOAT_TYPE, extended, contiguous);
   assert(status == PV_SUCCESS);

   getOutputFilename(path, "G_I", "_last");
   status = read(path, comm, &dtime, G_I, loc, PV_FLOAT_TYPE, extended, contiguous);
   assert(status == PV_SUCCESS);

   getOutputFilename(path, "G_IB", "_last");
   status = read(path, comm, &dtime, G_IB, loc, PV_FLOAT_TYPE, extended, contiguous);
   assert(status == PV_SUCCESS);

   getOutputFilename(path, "R", "_last");
   status = read(path, comm, &dtime, R, loc, PV_FLOAT_TYPE, extended, contiguous);
   assert(status == PV_SUCCESS);

   if(localWmaxFlag && Wmax != NULL){
      extended = true;
      getOutputFilename(path, "Wmax", "_last");
      status = read(path, comm, &dtime, Wmax, loc, PV_FLOAT_TYPE, extended, contiguous);
      assert(status == PV_SUCCESS);
   }

   *time = (float) dtime;
   return status;

}

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
   status = write(path, comm, time, Vth, loc, PV_FLOAT_TYPE, extended, contiguous);

   getOutputFilename(path, "G_E", last_str);
   status = write(path, comm, time, G_E, loc, PV_FLOAT_TYPE, extended, contiguous);

   getOutputFilename(path, "G_I", last_str);
   status = write(path, comm, time, G_I, loc, PV_FLOAT_TYPE, extended, contiguous);

   getOutputFilename(path, "G_IB", last_str);
   status = write(path, comm, time, G_IB, loc, PV_FLOAT_TYPE, extended, contiguous);

   getOutputFilename(path, "R", last_str);
   status = write(path, comm, time, R, loc, PV_FLOAT_TYPE, extended, contiguous);

   if(localWmaxFlag){
      extended = true;
      getOutputFilename(path, "Wmax", last_str);
      status = write(path, comm, time, Wmax, loc, PV_FLOAT_TYPE, extended, contiguous);
   }

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

#ifndef PV_USE_OPENCL
#  include "../kernels/LIF_update_state.cl"
#  include "../kernels/LIF_update_state_localWmax.cl"
#endif

#ifdef __cplusplus
}
#endif
