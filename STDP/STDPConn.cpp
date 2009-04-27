/*
 * RuleConn.cpp
 *
 *  Created on: Apr 5, 2009
 *      Author: rasmussn
 */

#include "STDPConn.hpp"
#include "src/io/io.h"
#include <assert.h>
#include <string.h>

namespace PV {

static int stdp_text_write_patch(FILE * fd, PVPatch * patch, float * data)
{
   int f, i, j;

   const int nx = (int) patch->nx;
   const int ny = (int) patch->ny;
   const int nf = (int) patch->nf;

   const int sx = (int) patch->sx;  assert(sx == nf);
   const int sy = (int) patch->sy;  //assert(sy == nf*nx);
   const int sf = (int) patch->sf;  assert(sf == 1);

   assert(fd != NULL);

   for (f = 0; f < nf; f++) {
      for (j = 0; j < ny; j++) {
         for (i = 0; i < nx; i++) {
            fprintf(fd, "%5.3f ", data[i*sx + j*sy + f*sf]);
         }
         //fprintf(fd, "\n");
      }
      //fprintf(fd, "\n");
   }

   return 0;
}


STDPConn::STDPConn(const char * name,
                   HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, int channel)
{
   this->connId = hc->numberOfConnections();
   this->name   = strdup(name);
   this->parent = hc;

   this->numBundles = 1;

   initialize(NULL, pre, post, channel);

   hc->addConnection(this);
}

int STDPConn::updateWeights(PVLayerCube * preActivityCube, int neighbor)
{
   const float dt = parent->getDeltaTime();
   const float decayLTP = expf(-dt/tauLTP);
//   const float decay    = expf(-dt/tauSTDP);

   // assume pDecr has been updated already, and weights have been used, so
   // 1. update Psij (pIncr) for each synapse
   // 2. update wij

   const int numActive = preActivityCube->numItems;

   // TODO - handle neighbors
   if (neighbor != 0) {
      return 0;
   }

   // TODO - this loop should be over all kPre not just active
   for (int kPre = 0; kPre < numActive; kPre++) {
      PVSynapseBundle * tasks = this->tasks(kPre, neighbor);

      const float preActivity = preActivityCube->data[kPre];

      for (unsigned int i = 0; i < tasks->numTasks; i++) {
         PVPatch * pIncr = tasks->tasks[i]->plasticIncr;
         PVPatch * w     = tasks->tasks[i]->weights;
         size_t offset   = tasks->tasks[i]->offset;

         float * postActivity = &post->clayer->activity->data[offset];
         float * M = &pDecr->data[offset];  // STDP decrement variable
         float * P =  pIncr->data;          // STDP increment variable

         int nk  = (int) pIncr->nf * (int) pIncr->nx;
         int ny  = (int) pIncr->ny;
         int sy  = (int) pIncr->sy;

         // TODO - unroll

         // update Psij (pIncr variable)
         for (int y = 0; y < ny; y++) {
            pvpatch_update_plasticity_incr(nk, pIncr->data + y*sy, preActivity, decayLTP, ampLTP);
         }

         // update weights
         for (int y = 0; y < ny; y++) {
            int yOff = y*sy;
            pvpatch_update_weights(nk, w->data + yOff, M + yOff, P + yOff,
                                   preActivity, postActivity + yOff, dWMax, wMax);
         }
      }
   }

   outputState(stdout, 576+6);
   outputState(stdout, 577+6);

   return 0;
}

int STDPConn::outputState(FILE * fp, int kPre)
{
   PVSynapseBundle * tasks = this->tasks(kPre, 0);
   PVPatch * pIncr = tasks->tasks[0]->plasticIncr;
   PVPatch * w     = tasks->tasks[0]->weights;
   size_t offset   = tasks->tasks[0]->offset;

   float * M = &pDecr->data[offset];  // STDP decrement variable

   fprintf(fp, "w%d:      M=", kPre);
   stdp_text_write_patch(fp, pIncr, M);
   fprintf(fp, "P=");
   stdp_text_write_patch(fp, pIncr, pIncr->data);  // write the P variable
   fprintf(fp, "w=");
   stdp_text_write_patch(fp, w, w->data);
   fprintf(fp, "\n");
   fflush(fp);

   return 0;
}

int STDPConn::initializeWeights(const char * filename)
{
   if (filename == NULL) {
      PVParams * params = parent->parameters();
      const float strength = params->value(name, "strength");

      const int xScale = pre->clayer->xScale;
      const int yScale = pre->clayer->yScale;

      int nfPre = pre->clayer->numFeatures;

      const int numPatches = numberOfWeightPatches();
      for (int i = 0; i < numPatches; i++) {
         int fPre = i % nfPre;
         ruleWeights(wPatches[i], fPre, xScale, yScale, strength);
      }
   }
   else {
      fprintf(stderr, "Initializing weights from a file not implemented for RuleConn\n");
      exit(1);
   } // end if for filename

   return 0;
}

int STDPConn::ruleWeights(PVPatch * wp, int fPre, int xScale, int yScale, float strength)
{
   pvdata_t * w = wp->data;

   const int nx = (int) wp->nx;
   const int ny = (int) wp->ny;
   const int nf = (int) wp->nf;

   // strides
   const int sx = (int) wp->sx;  assert(sx == nf);
   const int sy = (int) wp->sy;  assert(sy == nf*nx);
   const int sf = (int) wp->sf;  assert(sf == 1);

   assert(fPre >= 0 && fPre <= 1);
   assert(ny == 1);
   assert(nf == 1);

   // rule 16 (only 100 applies, left neighbor fires, I fire, all other patterns fire 0)
   // left (post view) -> right (pre view) -> 100 -> 000

   // loop over all post synaptic neurons in patch

   // initialize connections of OFF and ON cells to 0
   for (int f = 0; f < nf; f++) {
      for (int j = 0; j < ny; j++) {
         for (int i = 0; i < nx; i++) {
            w[i*sx + j*sy + f*sf] = 0;
         }
      }
   }

   // now set the actual pattern for rule 16 (0 0 0 1 0 0 0 0)

   // pre-synaptic neuron is an OFF cell
   if (fPre == 0) {
      for (int j = 0; j < ny; j++) {
         int f = 0;

         // OFF ON OFF pattern only
         w[0*sx + j*sy + f*sf] = .9;
         w[2*sx + j*sy + f*sf] = .9;

         continue;  // only one feature for now

         f = 0;
         // sub-rule 000 (first OFF cell fires)
         w[0*sx + j*sy + f*sf] = 1;
         w[1*sx + j*sy + f*sf] = 1;
         w[2*sx + j*sy + f*sf] = 1;

         // sub-rule 001 (second OFF cell fires)
         f = 2;
         w[1*sx + j*sy + f*sf] = 1;
         w[2*sx + j*sy + f*sf] = 1;

         // sub-rule 010 (third OFF cell fires)
         f = 4;
         w[0*sx + j*sy + f*sf] = 1;
         w[2*sx + j*sy + f*sf] = 1;

         // sub-rule 011 (fourth OFF cell fires)
         f = 6;
         w[2*sx + j*sy + f*sf] = 1;

         // sub-rule 100 (fifth _ON_ cell fires)
         f = 9;
         w[0*sx + j*sy + f*sf] = 1;
         w[1*sx + j*sy + f*sf] = 1;

         // sub-rule 101 (six OFF cell fires)
         f = 10;
         w[1*sx + j*sy + f*sf] = 1;

         // sub-rule 110 (seventh OFF cell fires)
         f = 12;
         w[0*sx + j*sy + f*sf] = 1;

         // sub-rule 111 (eighth OFF cell fires)
         f = 14;
      }
   }

   // pre-synaptic neuron is an ON cell
   if (fPre == 1) {
      for (int j = 0; j < ny; j++) {
         int f = 0;

         // OFF ON OFF pattern only
         w[1*sx + j*sy + f*sf] = .9;

         continue;  // only one feature for now

         // sub-rule 000 (first OFF cell fires)
         f = 0;

         // sub-rule 001 (second OFF cell fires)
         f = 2;
         w[0*sx + j*sy + f*sf] = 1;

         // sub-rule 010 (third OFF cell fires)
         f = 4;
         w[1*sx + j*sy + f*sf] = 1;

         // sub-rule 011 (fourth OFF cell fires)
         f = 6;
         w[0*sx + j*sy + f*sf] = 1;
         w[1*sx + j*sy + f*sf] = 1;

         // sub-rule 100 (fifth _ON_ cell fires)
         f = 9;
         w[2*sx + j*sy + f*sf] = 1;

         // sub-rule 101 (six OFF cell fires)
         f = 10;
         w[0*sx + j*sy + f*sf] = 1;
         w[2*sx + j*sy + f*sf] = 1;

         // sub-rule 110 (seventh OFF cell fires)
         f = 12;
         w[1*sx + j*sy + f*sf] = 1;
         w[2*sx + j*sy + f*sf] = 1;

         // sub-rule 111 (eighth OFF cell fires)
         f = 14;
         w[0*sx + j*sy + f*sf] = 1;
         w[1*sx + j*sy + f*sf] = 1;
         w[2*sx + j*sy + f*sf] = 1;
      }
   }

   for (int f = 0; f < nf; f++) {
      float factor = strength;
      for (int i = 0; i < nx*ny; i++) w[f + i*nf] *= factor;
   }

   return 0;
}

} // namespace PV
