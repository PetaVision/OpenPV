/*
 * LongRangeConn.cpp
 *
 *  Created on: Feb 23, 2009
 *      Author: rasmussn
 */

#include "LongRangeConn.hpp"
#include "../io/io.h"
#include <assert.h>
#include <string.h>

namespace PV {

LongRangeConn::LongRangeConn(const char * name,
      HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, int channel)
{
   this->connId = hc->numberOfConnections();
   this->name = strdup(name);
   this->parent = hc;

   this->numBundles = 1;

   initialize(NULL, pre, post, channel);

   hc->addConnection(this);
}

int LongRangeConn::initializeWeights(const char * filename)
{
   float aspect = 4.0;
   float sigma  = 2.0;
   float rMax   = 8.0;
   float lambda = sigma/0.8;    // gabor wavelength
   float strength = 1.0;

   PVParams * params = parent->parameters();

   rMax = sqrt(nxp * nxp + nyp * nyp);

   int noPost = 1;
   if (params->present(post->getName(), "no")) {
      noPost = (int) params->value(post->getName(), "no");
   }

   aspect = params->value(name, "aspect");
   sigma  = params->value(name, "sigma");
   lambda = params->value(name, "lambda");

   if (params->present(name, "rMax")) rMax = params->value(name, "rMax");
   if (params->present(name, "strength")) {
      strength = params->value(name, "strength");
   }

   const float r2Max = rMax * rMax;
   const int xScale = post->clayer->xScale - pre->clayer->xScale;
   const int yScale = post->clayer->xScale - pre->clayer->yScale;

   const int numPatches = numberOfWeightPatches();
   for (int k = 0; k < numPatches; k++) {
      const int bundle = 0;
      PVPatch * patch = getWeights(k, bundle);
      calcWeights(patch, k, noPost, xScale, yScale, aspect, sigma, r2Max, lambda, strength);
   }

   return 0;
}

/**
 * calculate synaptic weights for a patch
 */
int LongRangeConn::calcWeights(PVPatch * wp, int kPre, int no, int xScale, int yScale,
        float aspect, float sigma, float r2Max, float lambda, float strength)
{

   // TODO -FIXME - what about kPre??

   int numFlanks = 0;
   float shift = 0;
   float rotate = 1.0;

   pvdata_t * w = wp->data;

   const int nx = (int) wp->nx;
   const int ny = (int) wp->ny;
   const int nf = (int) wp->nf;

   // TODO - make sure this is correct
   if (nx * ny * nf == 0) return 1;

   const int sx = (int) wp->sx;  assert(sx == nf);
   const int sy = (int) wp->sy;  assert(sy == nf*nx);
   const int sf = (int) wp->sf;  assert(sf == 1);

   // const float dx = powf(2, xScale);
   // const float dy = powf(2, yScale);

   // pre-synaptic neuron is at the center of the patch (0,0)
   // (x0,y0) is at upper left corner of patch (i=0,j=0)
   // const float x0 = -(nx/2.0 - 0.5) * dx;
   // const float y0 = +(ny/2.0 - 0.5) * dy;

   /*
    * Need to get find xPre
    */
   const int nxPre = (int) pre->clayer->loc.nx;
   // pick one as center
   int ixPre = (int) ( nxPre * ( ((float) rand()) / (float) RAND_MAX) );

   if (ixPre < 3) ixPre = 3;
   if (ixPre > (nxPre - 1) - 3) ixPre = (nxPre - 1) - 3;

   for (int f = 0; f < nf; f++) {
      for (int j = 0; j < ny; j++) {
         for (int i = 0; i < nx; i++) {
            w[i*sx + j*sy + f*sf] = 0;
         }
//         int i = ixPre - 3;
//         w[(i++)*sx + j*sy + f*sf] = 0.2;
//         w[(i++)*sx + j*sy + f*sf] = 0.5;
//         w[(i++)*sx + j*sy + f*sf] = 0.8;
//         w[(i++)*sx + j*sy + f*sf] = 1.0;
//         w[(i++)*sx + j*sy + f*sf] = 0.8;
//         w[(i++)*sx + j*sy + f*sf] = 0.5;
//         w[(i++)*sx + j*sy + f*sf] = 0.2;
      }
   }

   return HyPerConn::gauss2DCalcWeights(wp, kPre, no, xScale, yScale,
                             numFlanks, shift, rotate,
                             aspect, sigma, r2Max, strength);

   // normalize
   for (int f = 0; f < nf; f++) {
      float sum = 0;
      for (int i = 0; i < nx*ny; i++) sum += w[f + i*nf];

      if (sum == 0.0) return 0;  // all weights == zero is ok

      float factor = strength/sum;
      for (int i = 0; i < nx*ny; i++) w[f + i*nf] *= factor;
   }

   return 0;
}

int LongRangeConn::createSynapseBundles(int numTasks)
{
   // TODO - these needs to be an input parameter obtained from the connection
   const float kxPost0Left = 0.0f;
   const float kyPost0Left = 0.0f;

   const float nxPre  = pre->clayer->loc.nx;
   const float nyPre  = pre->clayer->loc.ny;
   const float kx0Pre = pre->clayer->loc.kx0;
   const float ky0Pre = pre->clayer->loc.ky0;
   const float nfPre  = pre->clayer->numFeatures;

   const float nxPost  = post->clayer->loc.nx;
   const float nyPost  = post->clayer->loc.ny;
   const float kx0Post = post->clayer->loc.kx0;
   const float ky0Post = post->clayer->loc.ky0;
   const float nfPost  = post->clayer->numFeatures;

   const int xScale = post->clayer->xScale - pre->clayer->xScale;
   const int yScale = post->clayer->yScale - pre->clayer->yScale;

   const float numBorder = post->clayer->numBorder;
   assert(numBorder == 0);

#ifndef FEATURES_LAST
   const float psf = 1;
   const float psx = nfp;
   const float psy = psx * (nxPost + 2.0f*numBorder);
#else
   const float psx = 1;
   const float psy = nxPost + 2.0f*numBorder;
   const float psf = psy * (nyPost + 2.0f*numBorder);
#endif

   int nNeurons = pre->clayer->numNeurons;
   int nTotalTasks = nNeurons * numTasks;

   PVSynapseBundle ** allBPtrs   = (PVSynapseBundle**) malloc(nNeurons*sizeof(PVSynapseBundle*));
   PVSynapseBundle  * allBundles = (PVSynapseBundle *) malloc(nNeurons*sizeof(PVSynapseBundle));
   PVSynapseTask** allTPtrs = (PVSynapseTask**) malloc(nTotalTasks*sizeof(PVSynapseTask*));
   PVSynapseTask * allTasks = (PVSynapseTask*)  malloc(nTotalTasks*sizeof(PVSynapseTask));
   PVPatch       * allData  = (PVPatch*)        malloc(nTotalTasks*sizeof(PVPatch));

   bundles = allBPtrs;

   for (int kPre = 0; kPre < nNeurons; kPre++) {
      int offset = kPre*sizeof(PVSynapseBundle);
      bundles[kPre] = (PVSynapseBundle*) ((char*) allBundles + offset);
      bundles[kPre]->numTasks = numTasks;
      bundles[kPre]->tasks = allTPtrs + kPre*numTasks;

      PVSynapseBundle * list = bundles[kPre];
      for (int i = 0; i < numTasks; i++) {
         offset = (i + kPre*numTasks) * sizeof(PVSynapseTask);
         list->tasks[i] = (PVSynapseTask*) ((char*) allTasks + offset);
      }

      for (int i = 0; i < numTasks; i++) {
         PVSynapseTask * task = list->tasks[i];

         task->weights = this->getWeights(kPre, i);

         // global indices
         float kxPre = kx0Pre + kxPos(kPre, nxPre, nyPre, nfPre);
         float kyPre = ky0Pre + kyPos(kPre, nxPre, nyPre, nfPre);

         // Find random kxPre for given kPre
         kxPre = (int) (nxPre * ( ((float) rand()) / (float) RAND_MAX) + 0.5);

         // global indices
         float kxPost = pvlayer_patchHead(kxPre, kxPost0Left, xScale, nxp);
         float kyPost = pvlayer_patchHead(kyPre, kyPost0Left, yScale, nyp);

         // TODO - can get nf from weight patch but what about kf0?
         // weight patch is actually a pencil and so kfPost is always 0?
         float kfPost = 0.0f;

         // convert to local frame
         kxPost = kxPost - kx0Post;
         kyPost = kyPost - ky0Post;

         // adjust location so patch is in bounds
         float dx = 0;
         float dy = 0;
         float nxPatch = nxp;
         float nyPatch = nyp;

         if (kxPost < 0.0) {
            nxPatch -= -kxPost;
            kxPost = 0.0;
            if (nxPatch < 0.0) nxPatch = 0.0;
            dx = nxp - nxPatch;
         }
         else if (kxPost + nxp > nxPost) {
            nxPatch -= kxPost + nxp - nxPost;
            if (nxPatch < 0.0) {
               nxPatch = 0.0;
               kxPost  = nxPost - 1.0;
            }
         }

         if (kyPost < 0.0) {
            nyPatch -= -kyPost;
            kyPost = 0.0;
            if (nyPatch < 0.0) nyPatch = 0.0;
            dy = nyp - nyPatch;
         }
         else if (kyPost + nyp > nyPost) {
            nyPatch -= kyPost + nyp - nyPost;
            if (nyPatch < 0.0) {
               nyPatch = 0.0;
               kyPost  = nyPost - 1.0;
            }
         }

         // if out of bounds in x (y), also out in y (x)
         if (nxPatch == 0.0 || nyPatch == 0.0) {
            dx = 0.0;
            dy = 0.0;
            nxPatch = 0.0;
            nyPatch = 0.0;
         }

         // local index
         int kl = kIndex(kxPost, kyPost, kfPost, nxPost, nyPost, nfPost);
         assert(kl >= 0);
         assert(kl < post->clayer->numNeurons);

         pvdata_t * phi = post->clayer->phi[channel] + kl;

         // TODO - shouldn't kPre vary first?
//         offset = (i + kPre*numTasks) * sizeof(PVPatch);
         offset = (kPre + i*numTasks) * sizeof(PVPatch);
         task->data = (PVPatch*) ((char*) allData + offset);

         pvpatch_init(task->data, (int)nxPatch, (int)nyPatch, (int)nfp, psx, psy, psf, phi);
      }
   }

   return 0;
}

} // namespace PV
