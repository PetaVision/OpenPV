/*
 * Retina1DPattern.cpp
 *
 *  Created on: Apr 25, 2009
 *      Author: gkenyon
 */

#include "Retina1D.hpp"
#include <stdlib.h>
#include <assert.h>

namespace PV {

//Retina1D::Retina1D() : Retina() {
//}


Retina1D::Retina1D(const char * name, HyPerCol * hc) :
   Retina(name, hc)
{
   targ1D = (pvdata_t *) malloc(clayer->numNeurons * sizeof(pvdata_t));;
   createImage(clayer->V);
   for (int k = 0; k < clayer->numNeurons; k++) {
      targ1D[k] = 0.0;
   }
//   createRandomImage(clayer->V);
}

Retina1D::~Retina1D()
{
   if (targ1D) free(targ1D);
}

int Retina1D::createImage(pvdata_t * buf) {
   const int nx = clayer->loc.nx;
   const int ny = clayer->loc.ny;
   const int nf = clayer->numFeatures;

   int min0 = 1;
   int maxR = 15;

// traversing mom
   static int min = min0 - 4;
   static int max = min + 3;
   min += 2;
   max = min + 3;

   if (max > maxR) {
      min = min0;
      max = min + 3;
   }

// approaching mom
//   static int t = 0;
//   static int min = 11;  // not used first time through
//   static int max = 4;
//   if (t % 2) min -= 2;
//   else       max += 2;
//
//   t += 1;

   // slide image left and right by one pixel
//   t += 1;
//   min += t % 2;
//   max += t % 2;

   assert(this->clayer->numFeatures == 2);

   for (int k = 0; k < clayer->numNeurons; k += 2) {
      int kx = kxPos(k, nx, ny, nf);

      int pat = 1;
      if (kx < min || kx > max) {
         pat = 0;
      }
      buf[k]   = 1 - pat;
      buf[k+1] = pat;
   }

   return 0;
}

int Retina1D::createRandomImage(pvdata_t * buf) {
   // break into units of 3 pixels with 8 possible ON/OFF values

   int patterns[8][3];

   patterns[0][0] = 0;
   patterns[0][1] = 0;
   patterns[0][2] = 0;

   patterns[1][0] = 0;
   patterns[1][1] = 0;
   patterns[1][2] = 1;

   patterns[2][0] = 0;
   patterns[2][1] = 1;
   patterns[2][2] = 0;

   patterns[3][0] = 0;
   patterns[3][1] = 1;
   patterns[3][2] = 1;

   patterns[4][0] = 1;
   patterns[4][1] = 0;
   patterns[4][2] = 0;

   patterns[5][0] = 1;
   patterns[5][1] = 0;
   patterns[5][2] = 1;

   patterns[6][0] = 1;
   patterns[6][1] = 1;
   patterns[6][2] = 0;

   patterns[7][0] = 1;
   patterns[7][1] = 1;
   patterns[7][2] = 1;

   assert(this->clayer->numFeatures == 2);

   // TODO - 3 doesn't divide evenly into 64
//   for (int k = 0; k < clayer->numNeurons; k += 6) {
   for (int k = 0; k < clayer->numNeurons; k += 2) {
      //      int pat = rand() % 8;
//      assert(pat < 8);
//      buf[k+0] = patterns[pat][0];
//      buf[k+1] = patterns[pat][1];
//      buf[k+2] = patterns[pat][2];

      int pat = rand() % 2;
      buf[k]   = 1 - pat;
      buf[k+1] = pat;
   }

   return 0;
}

int Retina1D::updateState(float time, float dt)
{
   int start;

   fileread_params * params = (fileread_params *) clayer->params;
   float prob = params->poissonEdgeProb;

   pvdata_t * V = clayer->V;
   pvdata_t * activity = clayer->activity->data;

   for (int k = 0; k < clayer->numNeurons; k++) {
      int flag = spike(time, dt, prob, &start);

      if (start == 1) {  // start of a new burst cycle
         for (int kk = 0; kk < clayer->numNeurons; kk++) {
            targ1D[kk] = 0;
         }
         // create a new image for next time
         createImage(V);
      }

      if (targ1D[k] == 0 && flag) {
         activity[k] = V[k];
         targ1D[k] = 1;  // keep cell from firing again during burst period
      }
      else {
         activity[k] = 0;
      }
   }

   return 0;
}

/**
 * Returns 1 if an event should occur, 0 otherwise (let prob = 1 for nonspiking)
 */
int Retina1D::spike(float time, float dt, float prob, int * start)
{
   static int burstState = 0;

   fileread_params * params = (fileread_params *) clayer->params;
   float poissonProb  = RAND_MAX * prob;

   int burstStatus = 0;
   if (params->burstDuration <= 0 || params->burstFreq == 0) {
      burstStatus = sin( 2 * PI * time * params->burstFreq / 1000. ) >= 0.;
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

   if (stimStatus) return ( rand() < poissonProb );
   else            return 0;
}

}  // namespace PV
