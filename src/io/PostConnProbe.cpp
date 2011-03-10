/*
 * PostConnProbe.cpp
 *
 *  Created on: May 12, 2009
 *      Author: rasmussn
 */

#include "PostConnProbe.hpp"
#include "../layers/LIF.hpp"
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
   this->image = NULL;
   this->wPrev = NULL;
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
   this->image = NULL;
   this->wPrev = NULL;
   this->wActiv = NULL;
}

PostConnProbe::PostConnProbe(int kxPost, int kyPost, int kfPost)
   : ConnectionProbe(0)
{
   this->kxPost = kxPost;
   this->kyPost = kyPost;
   this->kfPost = kfPost;
   this->kPost = -1;
   this->image = NULL;
   this->wPrev = NULL;
   this->wActiv = NULL;
}

PostConnProbe::PostConnProbe(const char * filename,int kxPost, int kyPost, int kfPost)
   : ConnectionProbe(filename, 0, 0, 0)
{
   this->kxPost = kxPost;
   this->kyPost = kyPost;
   this->kfPost = kfPost;
   this->kPost = -1;
   this->image = NULL;
   this->wPrev = NULL;
   this->wActiv = NULL;
}

PostConnProbe::~PostConnProbe()
{
   if (wPrev  != NULL) free(wPrev);
   if (wActiv != NULL) free(wActiv);
}

/**
 * @time
 * @c
 * NOTES:
 *    - kPost, kxPost, kyPost are indices in the restricted post-synaptic layer.
 *
 */
int PostConnProbe::outputState(float time, HyPerConn * c)
{
   bool changed;
   int k, kxPre, kyPre;
   PVPatch  * w;
   PVPatch ** wPost = c->convertPreSynapticWeights(time);

   // TODO - WARNING: currently only works if nfPre==0

   const PVLayer * lPre = c->preSynapticLayer()->clayer;
   const PVLayer * lPost = c->postSynapticLayer()->clayer;
   // check if post is a LIF layer
   LIF * LIF_layer = dynamic_cast<LIF *>(c->postSynapticLayer());
   bool localWmaxFlag;
   if (LIF_layer != NULL){
      localWmaxFlag = LIF_layer->getLocalWmaxFlag();
   } else {
      localWmaxFlag = false;
   }

   const int nxPre = lPre->loc.nx;
   const int nyPre = lPre->loc.ny;
   const int nfPre = lPre->loc.nf;
   const int nbPre = lPre->loc.nb;

   const int nxPost = lPost->loc.nx;
   const int nyPost = lPost->loc.ny;
   const int nfPost = lPost->loc.nf;
   const int nbPost = lPost->loc.nb;

   // calc kPost if needed
   if (kPost < 0) {
      kPost = kIndex(kxPost, kyPost, kfPost, nxPost, nyPost, nfPost);
   }
   else {
      kxPost = kxPos(kPost, nxPost, nyPost, nfPost);
      kyPost = kyPos(kPost, nxPost, nyPost, nfPost);
      kfPost = featureIndex(kPost, nxPost, nyPost, nfPost);
   }

   c->preSynapticPatchHead(kxPost, kyPost, kfPost, &kxPre, &kyPre);

   const int kxPreEx = kxPre + nbPre;
   const int kyPreEx = kyPre + nbPre;

   const int kxPostEx = kxPost + nbPost;
   const int kyPostEx = kyPost + nbPost;
   const int kPostEx = kIndex(kxPostEx, kyPostEx, kfPost, nxPost+2*nbPost, nyPost+2*nbPost, nfPost);

   const bool postFired = lPost->activity->data[kPostEx] > 0.0;

   w = wPost[kPost];

   const int nw = w->nx * w->ny * w->nf;

   if (wPrev == NULL) {
      wPrev = (pvdata_t *) calloc(nw, sizeof(pvdata_t));
      for (k = 0; k < nw; k++) {
         wPrev[k] = w->data[k];
      }
   }
   if (wActiv == NULL) {
      wActiv = (pvdata_t *) calloc(nw, sizeof(pvdata_t));
   }

   k = 0;
   for (int ky = 0; ky < w->ny; ky++) {
      for (int kx = 0; kx < w->nx; kx++) {
         int kPre = kIndex(kx+kxPreEx, ky+kyPreEx, 0, nxPre+2*nbPre, nyPre+2*nbPre, nfPre);
         wActiv[k++] = lPre->activity->data[kPre];
      }
   }

   changed = false;
   for (k = 0; k < nw; k++) {
      if (wPrev[k] != w->data[k] || wActiv[k] != 0.0) {
         changed = true;
         break;
      }
   }
   if (stdpVars && (postFired || changed)) {
      if (postFired) fprintf(fp, "*");
      else fprintf(fp, " ");
      fprintf(fp, "t=%.1f w%d(%d,%d,%d) prePatchHead(%d,%d): ", time, kPost, kxPost,
            kyPost, kfPost, kxPre, kyPre);
      if (image) fprintf(fp, "tag==%d ", image->tag());
      fprintf(fp, "\n");
   }
   if (stdpVars && changed) {
      text_write_patch_extra(fp, w, w->data, wPrev, wActiv);
      fflush(fp);
   }

   for (k = 0; k < nw; k++) {
      wPrev[k] = w->data[k];
   }

   if (outputIndices) {
      fprintf(fp, "w%d(%d,%d,%d) prePatchHead(%d,%d): ", kPost, kxPost, kyPost, kfPost, kxPre, kyPre);
      if(!stdpVars){
        fprintf(fp,"\n");
      }
      const PVLayer * lPre = c->preSynapticLayer()->clayer;
      write_patch_indices(fp, w, &lPre->loc, kxPre, kyPre, 0);
      fflush(fp);
   }

   return 0;
}

int PostConnProbe::text_write_patch_extra(FILE * fp, PVPatch * patch,
                                          pvdata_t * data, pvdata_t * prev, pvdata_t * activ)
{
   int f, i, j;

   const int nx = patch->nx;
   const int ny = patch->ny;
   const int nf = patch->nf;

   const int sx = patch->sx;  assert(sx == nf);
   const int sy = patch->sy;  assert(sy == nf*nx); // stride could be weird at border
   const int sf = patch->sf;  assert(sf == 1);

   assert(fp != NULL);

   for (f = 0; f < nf; f++) {
      //fprintf(fp, "f = %i\n  ", f);
      for (j = 0; j < ny; j++) {
         for (i = 0; i < nx; i++) {
            int offset = i*sx + j*sy + f*sf;
            float diff = data[offset] - prev[offset];
            float a = activ[offset];

            const char * c = (a > 0.0) ? "*" : " ";

            if (diff == 0) {
               fprintf(fp, "%s%5.3f (     ) ", c, data[offset]);
            }
            else {
               fprintf(fp, "%s%5.3f (%+4.2f) ", c, data[offset], diff);
            }
         }
         //fprintf(fp, "\n  ");
      }
      fprintf(fp, "\n");
   }

   return 0;
}


} // namespace PV
