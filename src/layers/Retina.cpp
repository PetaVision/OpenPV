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

#ifdef OBSOLETE
Retina::Retina(const char * name, HyPerCol * hc, Image * img)
  : HyPerLayer(name, hc)
{
   this->img = img;
   initialize(TypeRetina);
}
#endif

#ifdef OBSOLETE
Retina::Retina(const char * name, HyPerCol * hc, const char * filename)
  : HyPerLayer(name, hc)
{
   this->img = new Image("Image", hc, filename);
   initialize(TypeRetina);
}
#endif

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

   // the size of the Retina may have changed due size of image
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

   return status;
}
//!
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

//! Updates the state of the Retina
/*!
 * REMARKS:
 *      - prevActivity[] buffer holds the time when a neuron last spiked.
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
   float probSpike = 0.0;
   fileread_params * params = (fileread_params *) clayer->params;

   pvdata_t * V = clayer->V;
   pvdata_t * phiExc   = clayer->phi[PHI_EXC];
   pvdata_t * phiInh   = clayer->phi[PHI_INH];
   pvdata_t * activity = clayer->activity->data;
   float    * prevActivity = clayer->prevActivity;

   const int nx = clayer->loc.nx;
   const int ny = clayer->loc.ny;
   const int nf = clayer->numFeatures;
   const int marginWidth = clayer->loc.nPad;

   #ifdef OBSOLETE
// should no longer need this as retina gets input from connection
//   updateImage(time, dt);
#endif OBSOLETE

   int numActive = 0;
   if (params->spikingFlag == 1) {
      const float probStim = params->poissonEdgeProb;   // need to multiply by V[k]
      const float probBase = params->poissonBlankProb;

      for (int k = 0; k < clayer->numNeurons; k++) {
         V[k] = phiExc[k] - phiInh[k];
         phiExc[k] = 0.0;
         phiInh[k] = 0.0;
      }

      for (int kex = 0; kex < clayer->numExtended; kex++) {
         float stimFactor = 0.0f;
         const float prevTime = prevActivity[kex];
         const int k = kIndexRestricted(kex, nx, ny, nf, marginWidth);
         int k_global;

         if (k > 0) {
            stimFactor = V[k];
         }

         activity[kex] = spike(time, dt, prevTime, probBase, probStim*stimFactor, &probSpike);
         prevActivity[kex] = (activity[kex] > 0.0) ? time : prevTime;
         if (k > 0 && activity[kex] > 0.0) {
#ifdef PV_USE_MPI
            k_global = globalIndexFromLocal(k, clayer->loc, nf);
#else
            k_global = k;
#endif
            clayer->activeIndices[numActive++] = k_global;
         }
      }
   }
   else {
      // retina is non spiking, pass scaled image through to activity
      //
      for (int k = 0; k < clayer->numNeurons; k++) {
         const int kex = kIndexExtended(k, nx, ny, nf, marginWidth);
         V[k] = phiExc[k] - phiInh[k];
         // scale output according to poissonEdgeProb, this could
         // perhaps be renamed when non spiking
         float maxRetinalActivity = params->poissonEdgeProb;
         activity[kex] = maxRetinalActivity * V[k];
         prevActivity[kex] = activity[kex];
         clayer->activeIndices[numActive++] = k;

         // reset accumulation buffers
         phiExc[k] = 0.0;
         phiInh[k] = 0.0;
      }
   }
   clayer->numActive = numActive;

#ifdef DEBUG_PRINT
   char filename[132];
   sprintf(filename, "r_%d.tiff", (int)(2*time));
   this->writeActivity(filename, time);
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

   if (params->burstDuration <= 0 || params->burstFreq == 0) {
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
