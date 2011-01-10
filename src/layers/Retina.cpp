/*
 * Retina.cpp
 *
 *  Created on: Jul 29, 2008
 *
 */

#include "HyPerLayer.hpp"
#include "Retina.hpp"
#include "../io/io.h"
#include "../include/default_params.h"
#include "../utils/cl_random.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void Retina_update_state (
    const float time,
    const float dt,
    const int nx,
    const int ny,
    const int nf,
    const int nb,
    Retina_params * params,
    uint4 * rnd,
    float * phiExc,
    float * phiInh,
    float * activity,
    float * prevTime);

#ifdef __cplusplus
}
#endif


namespace PV {

Retina::Retina(const char * name, HyPerCol * hc)
  : HyPerLayer(name, hc, NUM_RETINA_CHANNELS)
{
#ifdef OBSOLETE
   this->img = new Image("Image", hc, hc->inputFile());
#endif
   initialize(TypeRetina);
}

Retina::~Retina()
{
   free(rand_state);

#ifdef PV_USE_OPENCL
   free(evList);

   delete clPhiE;
   delete clPhiI;
   delete clActivity;
   delete clPrevTime;
#endif
}

int Retina::initialize(PVLayerType type)
{
   int status = 0;
   PVLayer * l = clayer;

   clayer->layerType = type;
   setParams(parent->parameters());

   // the size of the Retina may have changed due to size of image
   //
   const int nx = l->loc.nx;
   const int ny = l->loc.ny;
   const int nf = l->loc.nf;
   const int nb = l->loc.nb;
   l->numNeurons  = nx * ny * nf;
   l->numExtended = (nx + 2*nb) * (ny + 2*nb) * nf;

   // a random state variable is needed for every neuron/clthread
   rand_state = cl_random_init(l->numNeurons);

   status = parent->addLayer(this);

#ifdef PV_USE_OPENCL
   numEvents = NUM_RETINA_EVENTS;
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
int Retina::initializeThreadBuffers()
{
   int status = CL_SUCCESS;

   const size_t size    = clayer->numNeurons  * sizeof(pvdata_t);
   const size_t size_ex = clayer->numExtended * sizeof(pvdata_t);

   CLDevice * device = parent->getCLDevice();

   // these buffers are shared between host and device
   //

   clV = NULL;

   // TODO - use constant memory
   clParams = device->createBuffer(CL_MEM_COPY_HOST_PTR, sizeof(rParams), &rParams);
   clRand   = device->createBuffer(CL_MEM_COPY_HOST_PTR, getNumNeurons()*sizeof(uint4), rand_state);

   clPhiE = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, getChannel(CHANNEL_EXC));
   clPhiI = device->createBuffer(CL_MEM_COPY_HOST_PTR, size, getChannel(CHANNEL_INH));

   assert(numChannels < 3);
   clPhiIB = NULL;

   clActivity = device->createBuffer(CL_MEM_COPY_HOST_PTR, size_ex, clayer->activity->data);
   clPrevTime = device->createBuffer(CL_MEM_COPY_HOST_PTR, size_ex, clayer->prevActivity);

   return status;
}

int Retina::initializeThreadKernels()
{
   char kernelPath[256];
   char kernelFlags[256];

   int status = CL_SUCCESS;
   CLDevice * device = parent->getCLDevice();

   sprintf(kernelPath,  "%s/src/kernels/Retina_update_state.cl", parent->getPath());
   sprintf(kernelFlags, "-D PV_USE_OPENCL -I %s/src/kernels/",   parent->getPath());

   // create kernels
   //
   krUpdate = device->createKernel(kernelPath, "Retina_update_state", kernelFlags);

   int argid = 0;

   status |= krUpdate->setKernelArg(argid++, parent->simulationTime());
   status |= krUpdate->setKernelArg(argid++, parent->getDeltaTime());

   status |= krUpdate->setKernelArg(argid++, clayer->loc.nx);
   status |= krUpdate->setKernelArg(argid++, clayer->loc.ny);
   status |= krUpdate->setKernelArg(argid++, clayer->loc.nf);
   status |= krUpdate->setKernelArg(argid++, clayer->loc.nb);

   status |= krUpdate->setKernelArg(argid++, clParams);
   status |= krUpdate->setKernelArg(argid++, clRand);

   status |= krUpdate->setKernelArg(argid++, clPhiE);
   status |= krUpdate->setKernelArg(argid++, clPhiI);
   status |= krUpdate->setKernelArg(argid++, clActivity);
   status |= krUpdate->setKernelArg(argid++, clPrevTime);

   return status;
}
#endif

int Retina::setParams(PVParams * p)
{
   float dt_sec = parent->getDeltaTime() * .001;  // seconds

   clayer->loc.nf = 1;
   clayer->loc.nb = (int) p->value(name, "marginWidth", 0.0);

   clayer->params = &rParams;

   spikingFlag = (int) p->value(name, "spikingFlag", 1);

   float probStim = p->value(name, "poissonEdgeProb" , 1.0f);
   float probBase = p->value(name, "poissonBlankProb", 0.0f);

   if (p->present(name, "noiseOnFreq")) {
      probStim = p->value(name, "noiseOnFreq") * dt_sec;
      if (probStim > 1.0) probStim = 1.0;
   }
   if (p->present(name, "noiseOffFreq")) {
      probBase = p->value(name, "noiseOffFreq") * dt_sec;
      if (probBase > 1.0) probBase = 1.0;
   }

   // default parameters
   //
   rParams.probStim  = probStim;
   rParams.probBase  = probBase;
   rParams.beginStim = p->value(name, "beginStim", 0.0f);
   rParams.endStim   = p->value(name, "endStim"  , 99999999.9f);
   rParams.burstFreq = p->value(name, "burstFreq", 40);         // frequency of bursts
   rParams.burstDuration = p->value(name, "burstDuration", 20); // duration of each burst, <=0 -> sinusoidal
   rParams.refactory_period = REFACTORY_PERIOD;
   rParams.abs_refactory_period = REFACTORY_PERIOD;

   return 0;
}


int Retina::updateStateOpenCL(float time, float dt)
{
   int status = CL_SUCCESS;

   update_timer->start();

#ifdef PV_USE_OPENCL
   // wait for memory to be copied to device
   status |= clWaitForEvents(numWait, evList);
   for (int i = 1; i < numWait; i++) {
      clReleaseEvent(evList[i]);
   }
   numWait = 0;

   nxl = nyl = 1;

   status |= krUpdate->setKernelArg(0, time);
   status |= krUpdate->setKernelArg(1, dt);
   status |= krUpdate->run(getNumNeurons(), nxl*nyl, 0, NULL, &evUpdate);

   status |= clPhiE    ->copyFromDevice(1, &evUpdate, &evList[EV_R_PHI_E]);
   status |= clPhiI    ->copyFromDevice(1, &evUpdate, &evList[EV_R_PHI_I]);
   status |= clActivity->copyFromDevice(1, &evUpdate, &evList[EV_R_ACTIVITY]);

   numWait += 3;
#endif

   update_timer->start();

   return status;
}

int Retina::triggerReceive(InterColComm* comm)
{
   int status = HyPerLayer::triggerReceive(comm);

   // copy data to device
   //
#ifdef PV_USE_OPENCL
   status |= clPhiE->copyToDevice(&evList[EV_R_PHI_E]);
   status |= clPhiI->copyToDevice(&evList[EV_R_PHI_I]);
   numWait += 2;
#endif

   return status;
}

int Retina::waitOnPublish(InterColComm* comm)
{
   int status = HyPerLayer::waitOnPublish(comm);

   // copy activity to device
   //
#ifdef PV_USE_OPENCL
   status |= clActivity->copyToDevice(&evList[EV_R_ACTIVITY]);
   numWait += 1;
#endif

   return status;
}

//! Updates the state of the Retina
/*!
 * REMARKS:
 *      - prevActivity[] buffer holds the time when a neuron last spiked.
 *      - not used if nonspiking
 *      - it sets the probStim and probBase.
 *              - probStim = poissonEdgeProb * V[k];
 *              - probBase = poissonBlankProb
 *              .
 *      - activity[] is set to 0 or 1 depending on the return of spike()
 *      - this depends on the last time a neuron spiked as well as on V[]
 *      at the location of the neuron. This V[] is set by calling updateImage().
 *      - V points to the same memory space as data in the Image so that when Image
 *      is updated, V gets updated too.
 *      .
 * NOTES:
 *      - poissonEdgeProb = noiseOnFreq * dT
 *      - poissonBlankProb = noiseOffFreq * dT
 *      .
 *
 *
 */
int Retina::updateState(float time, float dt)
{
#ifndef PV_USE_OPENCL
   update_timer->start();

   const int nx = clayer->loc.nx;
   const int ny = clayer->loc.ny;
   const int nf = clayer->loc.nf;
   const int nb = clayer->loc.nb;

   pvdata_t * phiExc   = getChannel(CHANNEL_EXC);
   pvdata_t * phiInh   = getChannel(CHANNEL_INH);
   pvdata_t * activity = clayer->activity->data;

   if (spikingFlag == 1) {
      Retina_update_state(time, dt, nx, ny, nf, nb,
                          &rParams, rand_state,
                          phiExc, phiInh, activity, clayer->prevActivity);
   }
   else {
      // retina is non spiking, pass scaled image through to activity
      // scale by poissonEdgeProb (maxRetinalActivity)
      //
      for (int k = 0; k < clayer->numNeurons; k++) {
         const int kex = kIndexExtended(k, nx, ny, nf, nb);
         activity[kex] = rParams.probStim * (phiExc[k] - phiInh[k]);
         // reset accumulation buffers
         phiExc[k] = 0.0;
         phiInh[k] = 0.0;
      }
   }
   update_timer->stop();
#else

   updateStateOpenCL(time, dt);

#endif

#ifdef DEBUG_PRINT
   char filename[132];
   sprintf(filename, "r_%d.tiff", (int)(2*time));
   this->writeActivity(filename, time);

   printf("----------------\n");
   for (int k = 0; k < 6; k++) {
      printf("host:: k==%d h_exc==%f h_inh==%f\n", k, phiExc[k], phiInh[k]);
   }
   printf("----------------\n");

#endif

   return 0;
}

int Retina::writeState(const char * path, float time)
{
   HyPerLayer::writeState(path, time);

   // print activity at center of image

#ifdef DEBUG_OUTPUT
   int sx = clayer->numFeatures;
   int sy = sx*clayer->loc.nx;
   pvdata_t * a = clayer->activity->data;

   for (int k = 0; k < clayer->numExtended; k++) {
      if (a[k] == 1.0) printf("a[%d] == 1\n", k);
   }

  int n = (int) (sy*(clayer->loc.ny/2 - 1) + sx*(clayer->loc.nx/2));
  for (int f = 0; f < clayer->numFeatures; f++) {
     printf("a[%d] = %f\n", n, a[n]);
     n += 1;
  }
#endif

   return 0;
}

//! Spiking method for Retina
/*!
 * Returns 1 if an event should occur, 0 otherwise. This is a stochastic model.
 * REMARKS:
 *      - During ABS_REFACTORY_PERIOD a neuron does not spike
 *      - The neurons that correspond to stimuli (on Image pixels)
 *       spike with probability probStim.
 *      - The neurons that correspond to background image pixels
 *      spike with probability probBase.
 *      - After ABS_REFACTORY_PERIOD the spiking probability
 *        grows exponentially to probBase and probStim respectively.
 *      - The burst of the retina is periodic with period T set by
 *        T = 1000/burstFreq in miliseconds
 *      - When the time t is such that mT < t < mT + burstDuration, where m is
 *      an integer, the burstStatus is set to 1.
 *      - The burstStatus is also determined by the condition that
 *      beginStim < t < endStim. These parameters are set in the input
 *      params file params.stdp
 *      - sinAmp modulates the spiking probability only when burstDuration <= 0
 *      or burstFreq = 0
 *      - probSpike is set to probBase for all neurons.
 *      - for neurons exposed to Image on pixels, probSpike increases
 *       with probStim.
 *      - When the probability is negative, the neuron does not spike.
 *      .
 * NOTES:
 *      - time is measured in milliseconds.
 *      .
 */
#ifdef OBSOLETE
int Retina::spike(float time, float dt, float prev, float probBase, float probStim, float * probSpike)
{
   fileread_params * params = (fileread_params *) clayer->params;
   float burstStatus = 1;
   float sinAmp = 1.0;

   // see if neuron is in a refactory period
   //
   if ((time - prev) < ABS_REFACTORY_PERIOD) {
      return 0;
   }
   else {
      float delta = time - prev - ABS_REFACTORY_PERIOD;
      float refact = 1.0f - expf(-delta/REFACTORY_PERIOD);
      refact = (refact < 0) ? 0 : refact;
      probBase *= refact;
      probStim *= refact;
   }

   if (params->burstDuration < 0 || params->burstFreq == 0) {
      sinAmp = cos( 2 * PI * time * params->burstFreq / 1000. );
   }
   else {
      burstStatus = fmodf(time, 1000. / params->burstFreq);
      burstStatus = burstStatus <= params->burstDuration;
   }

   burstStatus *= (int) ( (time >= params->beginStim) && (time < params->endStim) );
   *probSpike = probBase;

   if ((int)burstStatus) {
      *probSpike += probStim * sinAmp;  // negative prob is OK
    }
   return ( pv_random_prob() < *probSpike );
}
#endif

} // namespace PV

///////////////////////////////////////////////////////
//
// implementation of Retina kernels
//

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PV_USE_OPENCL
#  include "../kernels/Retina_update_state.cl"
#endif

#ifdef __cplusplus
}
#endif
