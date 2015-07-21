/*
 * weights.cpp
 *
 *  Created on: May 1, 2009
 *      Author: rasmussn
 */

#include "../../include/pv_types.h"
#include "../io.h"
#include <assert.h>

#define NUM_PARAMS 6

int main(int argc, char * argv[])
{
   int err = 0;
   int numParams, nxPrePatch, nyPrePatch, nfPost;
   int params[NUM_PARAMS];

   assert(argc == 2);
   FILE * fp = pv_open_binary(argv[1], &numParams, &nxPrePatch, &nyPrePatch, &nfPost);
   assert(fp != NULL);
   assert(numParams == NUM_PARAMS);

   assert(pv_read_binary_params(fp, numParams, params) == numParams);

   float minVal = (float) params[3];
   float maxVal = (float) params[4];
   int   numPre = params[5];

   // allocate memory for patches from pre-synaptic neurons

   int numPrePatch = nxPrePatch * nyPrePatch * nfPost; // maximum number items in patch
   int patchSize = sizeof(PVPatch) + numPrePatch * sizeof(pvdata_t);

   PVPatch ** prePatches = (PVPatch **) malloc(numPre * sizeof(PVPatch *));
   unsigned char * prePatchBuf = (unsigned char *) malloc(numPre * patchSize);

   for (int i = 0; i < numPre; i++) {
      prePatches[i] = (PVPatch *) &prePatchBuf[i*patchSize];
      PVPatch * p = prePatches[i];
      p->data = (pvdata_t *) ((unsigned char *) p + sizeof(PVPatch));
   }

   err = pv_read_patches(fp, nfPost, minVal, maxVal, prePatches, numPre);

   PVPatch * p = prePatches[0];
   for (int i = 0; i < p->nx; i++) {
      printf("w[%d]=%f\n", i, p->data[i]);
   }


   int nxPre = 64;
   int nyPre = 1;
   int nfPre = 2;

   int nxPost = 64;
   int nyPost = 1;
   int numPost = nxPost * nyPost * nfPost;

   assert(nxPrePatch == 3);
   assert(nyPrePatch == 1);
   assert(numPre == nxPre * nyPre * nfPre);

   // TODO - this depends on scale
   int nxPostPatch = nxPrePatch;
   int nyPostPatch = nyPrePatch;

   // allocate memory for patches from post-synaptic neurons

   int numPostPatch = nxPostPatch * nyPostPatch * nfPre;
   patchSize = sizeof(PVPatch) + numPostPatch * sizeof(pvdata_t);

   PVPatch ** postPatches = (PVPatch **) malloc(numPost * sizeof(PVPatch *));
   unsigned char * postPatchBuf = (unsigned char *) malloc(numPost * patchSize);

   for (int i = 0; i < numPost; i++) {
      postPatches[i] = (PVPatch *) &postPatchBuf[i*patchSize];
      p = postPatches[i];
      p->nx = nxPostPatch;
      p->ny = nyPostPatch;
      p->nf = nfPre;
      p->sf = 1;
      p->sx = nfPre;
      p->sy = nfPre * nxPostPatch;
      p->data = (pvdata_t *) ((unsigned char *) p + sizeof(PVPatch));
   }

   // loop through post-synaptic neurons

   for (int kPost = 0; kPost < numPost; kPost++) {
      int kxPost = kxPos(kPost, nxPost, nyPost, nfPost);
      int kyPost = kyPos(kPost, nxPost, nyPost, nfPost);
      int kfPost = featureIndex(kPost, nxPost, nyPost, nfPost);
      int kxPreHead = kxPost - nxPostPatch/2;
      int kyPreHead = kyPost - nyPostPatch/2;

      for (int kp = 0; kp < numPostPatch; kp++) {
         int kxPostPatch = kxPos(kp, nxPostPatch, nyPostPatch, nfPre);
         int kyPostPatch = kyPos(kp, nxPostPatch, nyPostPatch, nfPre);
         int kfPostPatch = featureIndex(kp, nxPostPatch, nyPostPatch, nfPre);

         int kxPre = kxPreHead + kxPostPatch;
         int kyPre = kyPreHead + kyPostPatch;
         int kfPre = kfPostPatch;
         int kPre = kIndex(kxPre, kyPre, kfPre, nxPre, nyPre, nfPre);

         if (kPre < 0 || kPre >= nxPre*nyPre*nfPre) {
            assert(kxPre < 0 || kyPre < 0 || kxPre >= nxPre || kyPre >= nyPre);
            postPatches[kPost]->data[kp] = 0.0;
            //printf("kxPost=%2d kxPre=%2d kPre=%2d kp=%2d kxPrePatch=?  kxPostPatch=%2d kPrePatch=?  w=%f\n", kxPost, kxPre, kPre, kp, kxPostPatch, postPatches[kPost]->data[kp]);
         }
         else {
            p = prePatches[kPre];
            int nxp = (kxPre < nxPrePatch/2) ? p->nx : nxPrePatch;
            int nyp = (kyPre < nyPrePatch/2) ? p->ny : nyPrePatch;
            int kxPrePatch = nxp - (1 + kxPostPatch);
            int kyPrePatch = nyp - (1 + kyPostPatch);
            int kPrePatch = kIndex(kxPrePatch, kyPrePatch, kfPost, p->nx, p->ny, p->nf);

            postPatches[kPost]->data[kp] = p->data[kPrePatch];
            //printf("kxPost=%2d kxPre=%2d kPre=%2d kp=%2d kxPrePatch=%2d kxPostPatch=%2d kPrePatch=%2d w=%f\n", kxPost, kxPre, kPre, kp, kxPrePatch, kxPostPatch, kPrePatch, postPatches[kPost]->data[kp]);
         }
      }

   }

   pv_close_binary(fp);

   int append = 0;
   err = pv_write_patches("wx-postview", append, nxPostPatch, nyPostPatch, nfPre,
                          minVal, maxVal, numPost, postPatches);

   free(prePatchBuf);
   free(prePatches);
   free(postPatchBuf);
   free(postPatches);

   return err;
}
