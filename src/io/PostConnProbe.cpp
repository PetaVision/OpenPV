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
 * @filename
 * @conn
 * @kPost
 * @arbID
 */
PostConnProbe::PostConnProbe(const char * filename, HyPerConn * conn, int kPost, int arbID)
   : PatchProbe()
{
   initialize(NULL, filename, conn, INDEX_METHOD, 0, -1, -1, -1, arbID);
}

PostConnProbe::PostConnProbe(const char * filename, HyPerConn * conn, int kxPost, int kyPost, int kfPost, int arbID)
   : PatchProbe()
{
   initialize(NULL, filename, conn, COORDINATE_METHOD, -1, 0, 0, 0, arbID);
}

PostConnProbe::~PostConnProbe()
{
   if (wPrev  != NULL) free(wPrev);
   if (wActiv != NULL) free(wActiv);
}


int PostConnProbe::initialize(const char * probename, const char * filename, HyPerConn * conn, PatchIDMethod method, int kPost, int kxPost, int kyPost, int kfPost, int arbID) {
   int status = PatchProbe::initialize(NULL, filename, conn, method, kPost, kxPost, kyPost, kfPost, arbID);
   this->kxPost = kxPost;
   this->kyPost = kyPost;
   this->kfPost = kfPost;
   this->kPost = kPost;
   this->image = NULL;
   this->wPrev = NULL;
   this->wActiv = NULL;
   return status;
}
/**
 * @timef
 * NOTES:
 *    - kPost, kxPost, kyPost are indices in the restricted post-synaptic layer.
 *
 */
int PostConnProbe::outputState(double timef)
{
   int k, kxPre, kyPre;
   HyPerConn * c = getTargetConn();
   PVPatch  * w;
   PVPatch *** wPost = c->convertPreSynapticWeights(timef);

   // TODO - WARNING: currently only works if nfPre==0

   const PVLayer * lPre = c->preSynapticLayer()->clayer;
   const PVLayer * lPost = c->postSynapticLayer()->clayer;

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

   w = wPost[getArborID()][kPost];
   pvdata_t * wPostData = c->getWPostData(getArborID(),kPost);

   const int nw = w->nx * w->ny * nfPost; //w->nf;

   if (wPrev == NULL) {
      wPrev = (pvdata_t *) calloc(nw, sizeof(pvdata_t));
      for (k = 0; k < nw; k++) {
         wPrev[k] = wPostData[k]; // This is broken if the patch is shrunken
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

   bool changed = false;
   for (k = 0; k < nw; k++) {
      if (wPrev[k] != wPostData[k] || wActiv[k] != 0.0) {
         changed = true;
         break;
      }
   }
   FILE * fp = getFilePtr();
   if (stdpVars && (postFired || changed)) {
      if (postFired) fprintf(fp, "*");
      else fprintf(fp, " ");
      fprintf(fp, "t=%.1f w%d(%d,%d,%d) prePatchHead(%d,%d): ", timef, kPost, kxPost,
            kyPost, kfPost, kxPre, kyPre);
      if (image) fprintf(fp, "tag==%d ", image->tag());
      fprintf(fp, "\n");
   }
   if (stdpVars && changed) {
      text_write_patch_extra(fp, w, wPostData, wPrev, wActiv);
      fflush(fp);
   }

   for (k = 0; k < nw; k++) {
      wPrev[k] = wPostData[k];
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

//broken but never referenced: if used need to change patch size and strides to match post-synaptic patch
/*
int PostConnProbe::text_write_patch_extra(FILE * fp, PVPatch * patch,
                                          pvdata_t * data, pvdata_t * prev, pvdata_t * activ, HyPerConn * parentConn)
{
   int f, i, j;

   const int nx = patch->nx;
   const int ny = patch->ny;
   const int nf = parentConn->fPatchSize(); //patch->nf;

   const int sx = parentConn->xPatchStride(); //patch->sx;
   assert(sx == nfp);
   const int sy = parentConn->yPatchStride(); //patch->sy;
   assert(sy == nf*nxp); // stride could be weird at border
   const int sf = parentConn->fPatchStride(); //patch->sf;
   assert(sf == 1);

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
*/


} // namespace PV
