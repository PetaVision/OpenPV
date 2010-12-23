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
#include "../utils/pv_random.h"

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
    const float probStim,
    const float probBase,
    const int nx,
    const int ny,
    const int nf,
    const int nb,
    float * phi,
    float * activity,
    float * prevActivity);
#ifdef __cplusplus
}
#endif


namespace PV {

// default values
fileread_params RetinaParams =
{
   0.0, 0.0, 0.0, 0.0,
   0.0,
   0.0, 0.0,         /* burstFreg, burstDuration */
   0.0, 0.0, 0.0  /* marginWidth, beginStim, endStim */
};

Retina::Retina(const char * name, HyPerCol * hc)
  : HyPerLayer(name, hc)
{
#ifdef OBSOLETE
   this->img = new Image("Image", hc, hc->inputFile());
#endif
   initialize(TypeRetina);
}

int Retina::initialize(PVLayerType type)
{
   int status = 0;
   PVLayer  * l   = clayer;

   this->clayer->layerType = type;

   setParams(parent->parameters(), &RetinaParams);

   fileread_params * params = (fileread_params *) l->params;

#ifdef OBSOLETE
   l->loc = img->getImageLoc();
#endif
   l->loc.nPad = (int) params->marginWidth;
   l->loc.nBands = 1;

   // the size of the Retina may have changed due to size of image
   //
   const int nx = l->loc.nx;
   const int ny = l->loc.ny;
   const int nBorder = l->loc.nPad;
   l->numFeatures = l->loc.nBands;
   l->numNeurons  = nx * ny * l->numFeatures;
   l->numExtended = (nx + 2*nBorder) * (ny + 2*nBorder) * l->numFeatures;

#ifdef OBSOLETE
   PVParams * pvParams = parent->parameters();
   fireOffPixels = (int) pvParams->value(name, "fireOffPixels", 0);
#endif

   status = parent->addLayer(this);

#ifdef OBSOLETE
   // for the Retina, V is extended size, so resize
   if (l->numExtended != l->numNeurons) {
      l->numNeurons = l->numExtended;
      free(l->V);
      l->V = (pvdata_t *) calloc(l->numExtended, sizeof(float));
      assert(l->V != NULL);
      free(l->activeIndices);
      l->activeIndices = (unsigned int *) calloc(l->numNeurons, sizeof(unsigned int));
      assert(l->activeIndices != NULL);
   }
#endif

   // TODO - could free other layer parameters as they are not used

#ifdef OBSOLETE
   // use image's data buffer
   updateImage(parent->simulationTime(), parent->getDeltaTime());
   copyFromImageBuffer();

   pvdata_t * V = l->V;

   if (params->invert) {
      for (int k = 0; k < l->numExtended; k++) {
         V[k] = 1 - V[k];
      }
   }

   if (params->uncolor) {
      for (int k = 0; k < l->numExtended; k++) {
         V[k] = (V[k] == 0.0) ? 0.0 : 1.0;
      }
   }
#endif

#ifdef PV_USE_OPENCL
   initializeThreadBuffers();
   initializeThreadData();
   initializeThreadKernels();
#endif

   return status;
}

#ifdef PV_USE_OPENCL
int Retina::initializeThreadData()
{
   return CL_SUCCESS;
}

int Retina::initializeThreadKernels()
{
   int status = CL_SUCCESS;

   fileread_params * params = (fileread_params *) clayer->params;

   const float probStim = params->poissonEdgeProb;   // need to multiply by V[k]
   const float probBase = params->poissonBlankProb;

   // create kernels
   //
   updatestate_kernel = parent->getCLDevice()->createKernel("Retina_update_state.cl", "update_state");

   int argid = 0;

   status |= updatestate_kernel->setKernelArg(argid++, 0.0f); // time (changed by updateState)
   status |= updatestate_kernel->setKernelArg(argid++, 1.0f); // dt (changed by updateState)

   status |= updatestate_kernel->setKernelArg(argid++, probStim);
   status |= updatestate_kernel->setKernelArg(argid++, probBase);

   status |= updatestate_kernel->setKernelArg(argid++, clayer->loc.nx);
   status |= updatestate_kernel->setKernelArg(argid++, clayer->loc.ny);
   status |= updatestate_kernel->setKernelArg(argid++, clayer->numFeatures);
   status |= updatestate_kernel->setKernelArg(argid++, clayer->loc.nPad);

   status |= updatestate_kernel->setKernelArg(argid++, clBuffers.phi);
   status |= updatestate_kernel->setKernelArg(argid++, clBuffers.activity);
   status |= updatestate_kernel->setKernelArg(argid++, clBuffers.prevActivity);

   return status;
}
#endif

//! Sets the V data buffer
/*!
 *
 * dt is in seconds here!
 *
 */
int Retina::setParams(PVParams * params, fileread_params * p)
{
   float dt = parent->getDeltaTime() * .001;  // seconds

   clayer->params = (float *) malloc(sizeof(*p));
   assert(clayer->params != NULL);
   memcpy(clayer->params, p, sizeof(*p));

   clayer->numParams = sizeof(*p) / sizeof(float);

   fileread_params * cp = (fileread_params *) clayer->params;

   if (params->present(name, "invert"))  cp->invert  = params->value(name, "invert");
   if (params->present(name, "uncolor")) cp->uncolor = params->value(name, "uncolor");

   if (params->present(name, "spikingFlag"))
      cp->spikingFlag      = params->value(name, "spikingFlag");
   if (params->present(name, "poissonEdgeProb"))
      cp->poissonEdgeProb  = params->value(name, "poissonEdgeProb");
   if (params->present(name, "poissonBlankProb"))
      cp->poissonBlankProb = params->value(name, "poissonBlankProb");
   if (params->present(name, "burstFreq"))
      cp->burstFreq  = params->value(name, "burstFreq");
   if (params->present(name, "burstDuration"))
      cp->burstDuration  = params->value(name, "burstDuration");
   if (params->present(name, "marginWidth"))
      cp->marginWidth      = params->value(name, "marginWidth");

   if (params->present(name, "noiseOnFreq")) {
      cp->poissonEdgeProb  = params->value(name, "noiseOnFreq") * dt;
      if (cp->poissonEdgeProb > 1.0) cp->poissonEdgeProb = 1.0;
   }
   if (params->present(name, "noiseOffFreq")) {
      cp->poissonBlankProb  = params->value(name, "noiseOffFreq") * dt;
      if (cp->poissonBlankProb > 1.0) cp->poissonBlankProb = 1.0;
   }

   if (params->present(name, "beginStim")) cp->beginStim = params->value(name, "beginStim");
   if (params->present(name, "endStim"))   cp->endStim   = params->value(name, "endStim");

   return 0;
}

#ifdef OBSOLETE
//! Sets the V data buffer
/*!
 *
 * REMARKS:
 *      - this method is called from  updateImage()
 *      - copies from the Image data buffer into the V buffer
 *      - it normalizes the V buffer so that V <= 1.
 *      .
 *
 *
 */
int Retina::copyFromImageBuffer()
{
   const int nf = clayer->numFeatures;
   pvdata_t * V = clayer->V;

   PVLayerLoc imageLoc = img->getImageLoc();

   assert(clayer->loc.nx == imageLoc.nx && clayer->loc.ny == imageLoc.ny);

   pvdata_t * ibuf = img->getImageBuffer();

   // for now
   assert(nf == 1);

   // This is incorrect because V is extended
   // might be able to use image buffer directly
   //
   //HyPerLayer::copyToBuffer(V, ibuf, &imageLoc, true, 1.0f);

   // normalize so that V <= 1.0 (V in Retina is extended)
   pvdata_t vmax = 0;
   for (int k = 0; k < clayer->numExtended; k++) {
      V[k] = ibuf[k];
      vmax = ( V[k] > vmax ) ? V[k] : vmax;
   }
   if (vmax != 0){
      for (int k = 0; k < clayer->numExtended; k++) {
         V[k] = V[k] / vmax;
      }
   }
/*
   pvdata_t vmin = 0;
   for (int k = 0; k < clayer->numExtended; k++) {
      V[k] = ibuf[k];
      vmin = V[k] < vmin ? V[k] : vmin;
   }
   if (vmin < -1){
      for (int k = 0; k < clayer->numExtended; k++) {
         V[k] = V[k] / fabs(vmin);
      }
   }
*/

   //
   // otherwise handle OFF/ON cells

   // f[0] are OFF, f[1] are ON cells
//   const int count = imageLoc.nx * imageLoc.ny;
//   if (fireOffPixels) {
//      for (int k = 0; k < count; k++) {
//         V[2*k]   = 1 - ibuf[k] / Imax;
//         V[2*k+1] = ibuf[k] / Imax;
//      }
//   }
//   else {
//      for (int k = 0; k < count; k++) {
//         V[2*k]   = ibuf[k] / Imax;
//         V[2*k+1] = ibuf[k] / Imax;
//      }

   return 0;
}

//! updates the Image that Retina is exposed to
/*!
 *
 * REMARKS:
 *      - This depends on the Image class. The data buffer is generally modulated
 *      by the intensity at the image location.
 *
 *
 */
int Retina::updateImage(float time, float dt)
{
   bool changed = img->updateImage(time, dt);
   if (not changed) return 0;

   int status = copyFromImageBuffer();

   PVLayer  * l   = clayer;
   fileread_params * params = (fileread_params *) l->params;
   pvdata_t * V = clayer->V;

   if (params->invert) {
      for (int k = 0; k < l->numExtended; k++) {
         V[k] = 1 - V[k];
      }
   }

   if (params->uncolor) {
      for (int k = 0; k < l->numExtended; k++) {
         V[k] = (V[k] == 0.0) ? 0.0 : 1.0;
      }
   }

   return status;
}
#endif

#ifdef PV_USE_OPENCL
int Retina::updateStateOpenCL(float time, float dt)
{
   int status = CL_SUCCESS;

   // TODO - do this asynchronously
   status |= clBuffers.phi->copyToDevice();
   status |= clBuffers.activity->copyToDevice();

   // assume only time changes
   //
   status |= updatestate_kernel->setKernelArg(0, time);
   status |= updatestate_kernel->run(clayer->numNeurons, 64);

   // TODO - do this asynchronously
   status |= clBuffers.phi->copyFromDevice();
   status |= clBuffers.activity->copyFromDevice();

   return status;
}
#endif

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
   update_timer->start();

#ifndef PV_USE_OPENCL
   fileread_params * params = (fileread_params *) clayer->params;

   const float probStim = params->poissonEdgeProb;   // need to multiply by V[k]
   const float probBase = params->poissonBlankProb;

   const int nx = clayer->loc.nx;
   const int ny = clayer->loc.ny;
   const int nf = clayer->numFeatures;
   const int nb = clayer->loc.nPad;

   if (params->spikingFlag == 1) {
      pvdata_t * phi = clayer->phi[0];
      pvdata_t * activity = clayer->activity->data;
      float    * prevActivity = clayer->prevActivity;

      Retina_update_state(time, dt, probStim, probBase,
                          nx, ny, nf, nb,
                          phi, activity, prevActivity);

      // TODO
      // copy data from device
      //

      // TODO - move to halo exchange so don't have to wait for data
      // calculate active indices
      //

      int numActive = 0;
      for (int k = 0; k < clayer->numNeurons; k++) {
         const int kex = kIndexExtended(k, nx, ny, nf, nb);
         if (activity[kex] > 0.0) {
            clayer->activeIndices[numActive++] = globalIndexFromLocal(k, clayer->loc, nf);
         }
         clayer->numActive = numActive;
      }
   }
   else {
      pvdata_t * activity = clayer->activity->data;
      pvdata_t * phiExc   = clayer->phi[PHI_EXC];
      pvdata_t * phiInh   = clayer->phi[PHI_INH];

      // retina is non spiking, pass scaled image through to activity
      // scale by poissonEdgeProb (maxRetinalActivity)
      //
      for (int k = 0; k < clayer->numNeurons; k++) {
         const int kex = kIndexExtended(k, nx, ny, nf, nb);
         activity[kex] = probStim * (phiExc[k] - phiInh[k]);
         // TODO - get rid of this for performance
         clayer->activeIndices[k] = globalIndexFromLocal(k, clayer->loc, nf);

         // reset accumulation buffers
         phiExc[k] = 0.0;
         phiInh[k] = 0.0;
      }
      clayer->numActive = clayer->numNeurons;
   }
#else

   updateStateOpenCL(time, dt);

#endif

   update_timer->stop();


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

} // namespace PV

///////////////////////////////////////////////////////
//
// implementation of Retina kernels
//

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PV_USE_OPENCL
#  include "kernels/Retina_update_state.cl"
#endif

#ifdef __cplusplus
}
#endif


