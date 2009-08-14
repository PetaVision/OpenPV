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
   0.0, 1.0, 1.0, 1.0*(NOISE_AMP==0.0)+0.5*(NOISE_AMP>0.0),
   0.0*(NOISE_AMP==0.0)+0.01*(NOISE_AMP>0.0),
   0.0, 0.0,         /* burstFreg, burstDuration */
   0.0, 0.0, 1000.0, /* marginWidth, beginStim, endStim */
   NULL              /* filename */
};

Retina::Retina(const char * name, HyPerCol * hc)
  : HyPerLayer(name, hc)
{
   setParams(parent->parameters(), &RetinaParams, hc->inputFile());
   init(name, TypeRetina);
}

Retina::Retina(const char * name, HyPerCol * hc, const char * filename)
  : HyPerLayer(name, hc)
{
   if (filename == NULL) {
      filename = hc->inputFile();
   }
   setParams(parent->parameters(), &RetinaParams, filename);
   init(name, TypeRetina);
}

int Retina::init(const char * name, PVLayerType type)
{
   int n, err = 0;
   PVLayer  * l   = clayer;
   pvdata_t * buf = NULL;
   pvdata_t * V   = l->V;

   this->probes = NULL;
   this->ioAppend = 0;
   this->outputOnPublish = 1;
   this->clayer->layerType = type;

   this->numProbes = 0;

   fileread_params * params = (fileread_params *) l->params;
   float marginWidth = params->marginWidth;

   PVParams * pvParams = parent->parameters();

   int fireOffPixels = 0;
   if (pvParams->present(name, "fireOffPixels")) {
      fireOffPixels = pvParams->value(name, "fireOffPixels");
   }

   const int nx = l->loc.nx;
   const int ny = l->loc.ny;
   const int nf = l->numFeatures;

   buf = (pvdata_t *) malloc(nx*ny*nf*sizeof(pvdata_t));
   assert(buf != NULL);
   if (params->filename != NULL) {
      err = scatterReadFile(params->filename, l, buf, MPI_COMM_WORLD);
      if (err != 0) {
         free(buf);
         return err;
      }
   }
   else {
      err = createImage(buf);
      if (err != 0) {
         free(buf);
         return err;
      }
   }

   assert(nf == 1 || nf == 2);
   if (nf == 1) {
      for (int k = 0; k < l->numNeurons; k++) {
         V[k] = buf[k];
      }
   }
   else {
      // f[0] are OFF, f[1] are ON cells
      if (fireOffPixels) {
         for (int k = 0; k < nx*ny; k++) {
            V[2*k]   = 1 - buf[k];
            V[2*k+1] = buf[k];
         }
      }
      else {
         for (int k = 0; k < nx*ny; k++) {
            V[2*k]   = buf[k];
            V[2*k+1] = buf[k];
         }
      }
   }

   // check margins

   // TODO - make sure the origin information is working correctly
   if (marginWidth != 0.0f) {
      for (n = 0; n < l->numNeurons; n++) {
         float x = xPos(n, l->xOrigin, l->loc.dx, l->loc.nx, l->loc.ny, l->numFeatures);
         float y = yPos(n, l->yOrigin, l->loc.dy, l->loc.nx, l->loc.ny, l->numFeatures);
         if ( x < marginWidth || x > l->loc.nxGlobal * l->loc.dx - marginWidth ||
              y < marginWidth || y > l->loc.nyGlobal * l->loc.dy - marginWidth ) {
            clayer->V[n] = 0.0;
         }
      }
   }

   if (params->invert) {
      for (n = 0; n < l->numNeurons; n++) {
         V[n] = (V[n] == 0.0) ? 1.0 : 0.0;
      }
   }

   if (params->uncolor) {
      for (n = 0; n < l->numNeurons; n++) {
         V[n] = (V[n] == 0.0) ? 0.0 : 1.0;
      }
   }

   if (buf != NULL){
      free(buf);
   }
   // TODO - add retina boundary conditions

   return 0;
}

int Retina::setParams(PVParams * params, fileread_params * p, const char * filename)
{
   const char * name = getName();
   float dt = parent->getDeltaTime() * .001;  // seconds

   clayer->params = (float *) malloc(sizeof(*p));
   memcpy(clayer->params, p, sizeof(*p));

   clayer->numParams = sizeof(*p) / sizeof(float);
#ifdef PV_ARCH_64
   clayer->numParams -= 1; // extra space for (char *) filename pointer (pointer not same sizeof float)
#endif
   //assert(clayer->numParams == 9);

   fileread_params * cp = (fileread_params *) clayer->params;

   cp->filename = filename;

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

/*int Retina::updateState(float time, float dt)
{
   if (clayer->updateFunc != NULL) {
      clayer->updateFunc(clayer);
   }
   else {
      fileread_params * params = (fileread_params *) clayer->params;
      pvdata_t * activity = clayer->activity->data;
      float* V = clayer->V;

      float poissonEdgeProb  = RAND_MAX * params->poissonEdgeProb;
      float poissonBlankProb = RAND_MAX * params->poissonBlankProb;

      int burstStatus = 0;

      if (params->burstDuration <= 0 || params->burstFreq == 0) {
         burstStatus = sin( 2 * PI * time * params->burstFreq / 1000. ) > 0.;
      }
      else {
         burstStatus = fmod(time/dt, 1000. / (dt * params->burstFreq));
         burstStatus = burstStatus <= params->burstDuration;
      }

      int stimStatus = (time >= params->beginStim) && (time < params->endStim);
      stimStatus = stimStatus && burstStatus;

      if (params->spikingFlag == 0.0) {
         // non-spiking code
         if (stimStatus) {
            for (int k = 0; k < clayer->numNeurons; k++) {
               activity[k] = V[k];
            }
         }
         else {
            for (int k = 0; k < clayer->numNeurons; k++) {
               activity[k] = 0.0;
            }
         }
      }
      else {
         // Poisson spiking...
         const int nf = clayer->numFeatures;
         const int numNeurons = clayer->numNeurons;

         assert(nf > 0 && nf < 3);

         if (stimStatus == 0) {
            // fire at the background rate
            for (int k = 0; k < numNeurons; k++) {
               activity[k] = rand() < poissonBlankProb;
            }
         }
         else {
            // ON case (k even)
            for (int k = 0; k < clayer->numNeurons; k += nf) {
               if ( V[k] == 0.0 )
                  // fire at the background rate
                  activity[k] = (rand() < poissonBlankProb );
               else if ( V[k] > 0.0 )
                  // for gray scale use poissonEdgeProb * abs( V[k] )
                  activity[k] = (rand() < poissonEdgeProb );
               else // V[k] < 0.0
                  // fire at the below background rate (treated as zero if P < 0)
                  activity[k] = (rand() < ( 2 * poissonBlankProb - poissonEdgeProb ) );
            }
            // OFF case (k is odd)
            if (nf == 2) {
               for (int k = 1; k < clayer->numNeurons; k += nf) {
                   if ( V[k] == 0.0 )
                      // fire at the background rate
                      activity[k] = (rand() < poissonBlankProb );
                   else if ( V[k] < 0.0 )
                      // for gray scale use poissonEdgeProb * abs( V[k] )
                      activity[k] = (rand() < poissonEdgeProb );
                   else // V[k] > 0.0
                      // fire at the below background rate (treated as zero if P < 0)
                      activity[k] = (rand() < ( 2 * poissonBlankProb - poissonEdgeProb ) );
                }
            } // nf == 2
         } // stimStatus
      }
   }

   return 0;
} */

int Retina::updateState(float time, float dt)
{
   int start;

   fileread_params * params = (fileread_params *) clayer->params;

   pvdata_t * V = clayer->V;
   pvdata_t * activity = clayer->activity->data;

   for (int k = 0; k < clayer->numNeurons; k++) {
      float probStim = params->poissonEdgeProb * V[k];
      float prob = params->poissonBlankProb;

      int flag = spike(time, dt, prob, probStim, &start);
      activity[k] = (flag) ? 1.0 : 0.0;

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

   for (int i = 0; i < clayer->numNeurons; i++) {
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

int Retina::createImage(pvdata_t * buf)
{
   for (int i = 0; i < clayer->numNeurons; i++) {
      buf[i] = 0.0;
   }
   return 0;
}

/**
 * Returns 1 if an event should occur, 0 otherwise (let prob = 1 for nonspiking)
 */
int Retina::spike(float time, float dt, float prob, float probStim, int * start)
{
   static int burstState = 0;

   fileread_params * params = (fileread_params *) clayer->params;
   float poissonProb  = RAND_MAX * prob;
   probStim = RAND_MAX * probStim;


   int burstStatus = 0;
   float sinAmp = 1.0;

   if (params->burstDuration <= 0 || params->burstFreq == 0) {
      sinAmp = sin( 2 * PI * time * params->burstFreq / 1000. );
      burstStatus = sinAmp >= 0.;
   }
   else {
      burstStatus = fmod(time/dt, 1000. / (dt * params->burstFreq));
      burstStatus = burstStatus <= params->burstDuration;
   }

   *start = 0;
   if (burstState == 0 && burstStatus == 1) {
      *start = 1;
      burstState = 1;
   }
   else if (burstState == 1 && burstStatus == 0) {
      burstState = 0;
   }

   int stimStatus = (time >= params->beginStim) && (time < params->endStim);

   stimStatus = stimStatus && burstStatus;

   if (stimStatus) {
      poissonProb = (2 * poissonProb) + (probStim * sinAmp);
      return ( rand() < poissonProb );
   }
   else {
      if (sinAmp <= 0)   poissonProb = (poissonProb * sinAmp) + poissonProb;
      return ( rand() < poissonProb );
   }
/*
 * This is if you wish to go back to on/off instead of sinusoidal
   if (stimStatus) return (rand() < poissonProb);
   else return 0;
*/
}

}
