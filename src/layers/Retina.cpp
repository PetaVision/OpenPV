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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace PV {

// default values
fileread_params RetinaParams =
{
   0.0, 0.0, 1.0, 1.0*(NOISE_AMP==0.0)+0.5*(NOISE_AMP>0.0),
   0.0*(NOISE_AMP==0.0)+0.01*(NOISE_AMP>0.0),
   0.0, 0.0,         /* burstFreg, burstDuration */
   0.0, 0.0, 1000.0  /* marginWidth, beginStim, endStim */
};

Retina::Retina(const char * name, HyPerCol * hc)
  : HyPerLayer(name, hc)
{
   this->img = new Image("Image", hc, hc->inputFile());
   initialize(TypeRetina);
}

Retina::Retina(const char * name, HyPerCol * hc, Image * img)
  : HyPerLayer(name, hc)
{
   this->img = img;
   initialize(TypeRetina);
}

Retina::Retina(const char * name, HyPerCol * hc, const char * filename)
  : HyPerLayer(name, hc)
{
   this->img = new Image("Image", hc, filename);
   initialize(TypeRetina);
}

int Retina::initialize(PVLayerType type)
{
   int n, status = 0;
   PVLayer  * l   = clayer;

   this->clayer->layerType = type;

   setParams(parent->parameters(), &RetinaParams);

   fileread_params * params = (fileread_params *) l->params;

   l->loc = img->getImageLoc();
   l->loc.nPad   = params->marginWidth;
   l->loc.nBands = 1;

   PVParams * pvParams = parent->parameters();

   fireOffPixels = 0;
   if (pvParams->present(name, "fireOffPixels")) {
      fireOffPixels = pvParams->value(name, "fireOffPixels");
   }

   status = parent->addLayer(this);

   // at least for the Retina, V is extended size, so resize
   if (l->numExtended != l->numNeurons) {
      free(l->V);
      l->V = (pvdata_t *) calloc(l->numExtended, sizeof(float));
   }

   // TODO - could free other layer parameters as they are not used

   // use image's data buffer
   updateImage(parent->simulationTime(), parent->getDeltaTime());
   copyFromImageBuffer();

   // check margins/border region

   pvdata_t * V = l->V;

//   const int nxBorder = l->loc.nPad;
//   const int nyBorder = l->loc.nPad;

   // TODO - make sure the origin information is working correctly
   // TODO - I don't think this is needed any longer
//   if (nxBorder != 0.0f || nyBorder != 0.0f) {
//      for (n = 0; n < l->numExtended; n++) {
//         float x = xPos(n, l->xOrigin, l->dx, l->loc.nx, l->loc.ny, l->numFeatures);
//         float y = yPos(n, l->yOrigin, l->dy, l->loc.nx, l->loc.ny, l->numFeatures);
//         if ( x < nxBorder || x > l->loc.nxGlobal * l->dx - nxBorder ||
//              y < nyBorder || y > l->loc.nyGlobal * l->dy - nyBorder ) {
//            clayer->V[n] = 0.0;
//         }
//      }
//   }

   if (params->invert) {
      for (n = 0; n < l->numExtended; n++) {
         V[n] = (V[n] == 0.0) ? 1.0 : 0.0;
      }
   }

   if (params->uncolor) {
      for (n = 0; n < l->numExtended; n++) {
         V[n] = (V[n] == 0.0) ? 0.0 : 1.0;
      }
   }

   return status;
}

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

int Retina::recvSynapticInput(HyPerLayer* lSource, PVLayerCube* cube)
{
   return 0; //PV_Retina_recv_synaptic_input();
}


int Retina::copyFromImageBuffer()
{
   const int nf = clayer->numFeatures;
   pvdata_t * V = clayer->V;

   LayerLoc imageLoc = img->getImageLoc();

   assert(clayer->loc.nx == imageLoc.nx && clayer->loc.ny * imageLoc.ny);

   pvdata_t * ibuf = img->getImageBuffer();

   // normalize so that V <= 1.0
   pvdata_t Imax = 0;
   for (int k = 0; k < clayer->numExtended; k++) {
      Imax = ibuf[k] > Imax ? ibuf[k] : Imax;
   }
   if (Imax == 0){
      Imax = 1.0; // avoid divide by zero
   }


   // for now
   assert(nf == 1);
   if (nf == 1) {
      for (int k = 0; k < clayer->numExtended; k++) {
         V[k] = ibuf[k] / Imax;
      }
   }
   else {
      // f[0] are OFF, f[1] are ON cells
      const int count = imageLoc.nx * imageLoc.ny;
      if (fireOffPixels) {
         for (int k = 0; k < count; k++) {
            V[2*k]   = 1 - ( ibuf[k] / Imax );
            V[2*k+1] = ibuf[k] / Imax;
         }
      }
      else {
         for (int k = 0; k < count; k++) {
            V[2*k]   = ibuf[k] / Imax;
            V[2*k+1] = ibuf[k] / Imax;
         }
      }
   }

   return 0;
}

int Retina::updateImage(float time, float dt)
{
   bool changed = img->updateImage(time, dt);
   if (not changed) return 0;

   return copyFromImageBuffer();
}

int Retina::updateState(float time, float dt)
{
   float probSpike = 0.0;
   fileread_params * params = (fileread_params *) clayer->params;

   pvdata_t * V = clayer->V;
   pvdata_t * activity = clayer->activity->data;

   updateImage(time, dt);

   if (params->spikingFlag == 1) {
      for (int k = 0; k < clayer->numExtended; k++) {
         float probStim = params->poissonEdgeProb * V[k];
         float probBase = params->poissonBlankProb;
         activity[k] = spike(time, dt, probBase, probStim, &probSpike);
      }
   }
   else {
      for (int k = 0; k < clayer->numExtended; k++) {
         float probStim = params->poissonEdgeProb * V[k];
         float probBase = params->poissonBlankProb;
         spike(time, dt, probBase, probStim, &probSpike);
         activity[k] = probSpike;
      }
   }

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

   for (int i = 0; i < clayer->numExtended; i++) {
      if (a[i] == 1.0) printf("a[%d] == 1\n", i);
   }

  int n = (int) (sy*(clayer->loc.ny/2 - 1) + sx*(clayer->loc.nx/2));
  for (int f = 0; f < clayer->numFeatures; f++) {
     printf("a[%d] = %f\n", n, a[n]);
     n += 1;
  }
#endif

   return 0;
}



/**
 * Returns 1 if an event should occur, 0 otherwise (let prob = 1 for nonspiking)
 */
int Retina::spike(float time, float dt, float probBase, float probStim, float * probSpike)
{
   fileread_params * params = (fileread_params *) clayer->params;
   int burstStatus = 1;
   float sinAmp = 1.0;

   if (params->burstDuration <= 0 || params->burstFreq == 0) {
      sinAmp = cos( 2 * PI * time * params->burstFreq / 1000. );
   }
   else {
      burstStatus = fmod(time/dt, 1000. / (dt * params->burstFreq));
      burstStatus = burstStatus <= params->burstDuration;
   }

   burstStatus *= (int) ( (time >= params->beginStim) && (time < params->endStim) );
   *probSpike = probBase;

   if (burstStatus) {
      *probSpike += probStim * sinAmp;  // negative prob is OK
    }
   return ( rand() < (*probSpike * RAND_MAX));
}

}
