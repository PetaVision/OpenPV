/*
 * LinearPostConnProbe.cpp
 *
 *  Created on: May 12, 2009
 *      Author: rasmussn
 */

#include "LinearPostConnProbe.hpp"
#include <assert.h>

namespace PV {

LinearPostConnProbe::LinearPostConnProbe(PVDimType dim, int loc, int f)
   : PostConnProbe(0)
{
   this->dim = dim;
   this->loc = loc;
   this->f   = f;
}

LinearPostConnProbe::LinearPostConnProbe(int kPost)
   : PostConnProbe(0)
{
   this->kPost = kPost;
}

LinearPostConnProbe::LinearPostConnProbe(const char * filename, int kPost)
   : PostConnProbe(filename, 0)
{
   this->kPost = kPost;
}

int LinearPostConnProbe::outputState(float time, HyPerConn * conn)
{
   float min = conn->minWeight() + 0.5;
   float max = 0.98 * conn->maxWeight();

   assert(dim == DimX);

   int kyPost = 0;
   int kfPost = this->f;

   HyPerLayer * post = conn->postSynapticLayer();

   int nxPost = post->clayer->loc.nx;
   int nyPost = post->clayer->loc.ny;
   int nfPost = post->clayer->loc.nf;

   PVPatch ** wPost = conn->convertPreSynapticWeights(time);

   fprintf(fp, "       post-synaptic w state :");

   float rowSum = 0.0;
   for (int kx = 0; kx < nxPost; kx++) {
      char c = ' ';
      int kPost = kIndex(kx, kyPost, kfPost, nxPost, nyPost, nfPost);
      PVPatch * w = wPost[kPost];

      float w00 = w->data[0 + 0 * (int)w->sx];
      float w01 = w->data[0 + 1 * (int)w->sx];
      float w10 = w->data[1 + 0 * (int)w->sx];
      float w11 = w->data[1 + 1 * (int)w->sx];

#ifdef THREE
      float w00 = w->data[0 + 0 * (int)w->sx];
      float w01 = w->data[0 + 1 * (int)w->sx];
      float w02 = w->data[0 + 2 * (int)w->sx];
      float w10 = w->data[1 + 0 * (int)w->sx];
      float w11 = w->data[1 + 1 * (int)w->sx];
      float w12 = w->data[1 + 2 * (int)w->sx];
#endif

#ifndef AVERAGE_WEIGHTS
      if (w00 > max && w01 > max) c = '0';
      if (w10 > max && w11 > max) c = '1';
      if (w00 > max && w11 > max) c = 'l';
      if (w10 > max && w01 > max) c = 'r';

#ifdef THREE
      if (w00 > max && w01 > max && w12 > max) c = '1';
      if (w00 > max && w11 > max && w02 > max) c = '2';
      if (w00 > max && w11 > max && w12 > max) c = '3';
      if (w10 > max && w01 > max && w02 > max) c = '4';
      if (w10 > max && w01 > max && w12 > max) c = '5';
      if (w10 > max && w11 > max && w02 > max) c = '6';
// preferentially report '0' & '7'
      if (w00 > max && w01 > max && w02 > max) c = '0';
      if (w10 > max && w11 > max && w12 > max) c = '7';
#endif

      fprintf(fp, "%c", c);

#else
      float sum = w00 + w01 + w02 + w10 + w11 + w12;
      sum = (sum/conn->maxWeight()) * 10.0/6.0;
      rowSum += sum / nxPost;

      fprintf(fp, "%1d", (int) sum);
#endif
   }

   fprintf(fp, ":  %f\n", rowSum);
   fflush(fp);

   return 0;
}

} // namespace PV
