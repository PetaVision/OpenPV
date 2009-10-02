/*
 * RuleConn.cpp
 *
 *  Created on: Apr 5, 2009
 *      Author: rasmussn
 */

#include "RuleConn.hpp"
#include <assert.h>
#include <string.h>

namespace PV {

RuleConn::RuleConn(const char * name,
                   HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, int channel)
{
   initialize_base();
   initialize(name, hc, pre, post, channel, NULL);
}

int RuleConn::initializeWeights(const char * filename)
{
   if (filename == NULL) {
      PVParams * params = parent->parameters();
      const float strength = params->value(name, "strength");

      const int xScale = pre->clayer->xScale;
      const int yScale = pre->clayer->yScale;

      int nfPre = pre->clayer->numFeatures;

      const int arbor = 0;
      const int numPatches = numWeightPatches(arbor);
      for (int i = 0; i < numPatches; i++) {
         int fPre = i % nfPre;
         ruleWeights(wPatches[arbor][i], fPre, xScale, yScale, strength);
      }
   }
   else {
      fprintf(stderr, "Initializing weights from a file not implemented for RuleConn\n");
      exit(1);
   } // end if for filename

   return 0;
}

int RuleConn::ruleWeights(PVPatch * wp, int fPre, int xScale, int yScale, float strength)
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
   assert(nf == 16);

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
         // sub-rule 000 (first OFF cell fires)
         int f = 0;
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
         // sub-rule 000 (first OFF cell fires)
         int f = 0;

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
