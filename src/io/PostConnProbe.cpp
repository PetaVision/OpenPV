/*
 * PostConnProbe.cpp
 *
 *  Created on: May 12, 2009
 *      Author: rasmussn
 */

#include "PostConnProbe.hpp"
#include <assert.h>

namespace PV {

/**
 * @kPost
 */
PostConnProbe::PostConnProbe(int kPost)
   : ConnectionProbe(0)
{
   this->kxPost = 0;
   this->kyPost = 0;
   this->kfPost = 0;
   this->kPost = kPost;
}

/**
 * @filename
 * @kPost
 */
PostConnProbe::PostConnProbe(const char * filename, int kPost)
   : ConnectionProbe(filename, 0)
{
   this->kxPost = 0;
   this->kyPost = 0;
   this->kfPost = 0;
   this->kPost = kPost;
   this->outputIndices = false;
}

PostConnProbe::PostConnProbe(int kxPost, int kyPost, int kfPost)
   : ConnectionProbe(0)
{
   this->kxPost = kxPost;
   this->kyPost = kyPost;
   this->kfPost = kfPost;
   this->kPost = -1;
   this->outputIndices = false;
}

/**
 * @time
 * @c
 */
int PostConnProbe::outputState(float time, HyPerConn * c)
{
   int kxPre, kyPre;
   PVPatch * w;
   PVPatch ** wPost = c->convertPreSynapticWeights(time);

   const PVLayer * l = c->postSynapticLayer()->clayer;

   const float nx = l->loc.nx;
   const float ny = l->loc.ny;
   const float nf = l->numFeatures;

   // calc kPost if needed
   if (kPost < 0) {
      kPost = kIndex((float) kxPost, (float) kyPost, (float) kfPost, nx, ny, nf);
   }

   c->preSynapticPatchHead(kxPost, kyPost, kfPost, &kxPre, &kyPre);

   w = wPost[kPost];

   fprintf(fp, "w%d(%d,%d,%d) prePatchHead(%d,%d): ", kPost, kxPost, kyPost, kfPost, kxPre, kyPre);

   text_write_patch(fp, w, w->data);
   fflush(fp);

   if (outputIndices) {
      write_patch_indices(fp, w, &l->loc, kxPre, kyPre, 0);
      fflush(fp);
   }

   return 0;
}

} // namespace PV
