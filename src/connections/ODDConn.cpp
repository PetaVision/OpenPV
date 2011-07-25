/*
 * ODDConn.cpp
 *
 *  Created on: ?
 *      Author: kenyon
 */

#include "ODDConn.hpp"
#include "../io/io.h"
#include "../utils/conversions.h"
#include "../include/pv_types.h"
#include <assert.h>
#include <string.h>
#include <float.h>

namespace PV {

ODDConn::ODDConn()
{
   printf("GeislerConn::GeislerConn: running default constructor\n");
   initialize_base();
}

ODDConn::ODDConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
                         HyPerLayer * post, ChannelType channel)
{
   initialize_base();
   initialize(name, hc, pre, post, channel, NULL);
   // constructWeights(NULL); // HyPerConn::constructWeights moved back into HyPerConn::initialize
}

#ifdef OBSOLETE // marked obsolete Jul 21, 2011.  No routine calls it, and it doesn't make sense to define a connection without specifying a channel.
ODDConn::ODDConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post)
{
   initialize_base();
   initialize(name, hc, pre, post, CHANNEL_EXC, NULL); // use default channel
   constructWeights(NULL);
}
#endif

// provide filename or set to NULL
ODDConn::ODDConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, ChannelType channel, const char * filename)
{
   initialize_base();
   initialize(name, hc, pre, post, channel, filename);
   // constructWeights(filename); // HyPerConn::constructWeights moved back into HyPerConn::initialize
}

int ODDConn::initialize_base()
{
   numUpdates = 0;
   avePreActivity = NULL;
   avePostActivity = NULL;
   geislerPatches = NULL;
   return PV_SUCCESS; // return KernelConn::initialize_base();
}


PVPatch ** ODDConn::createWeights(PVPatch ** patches, int nPatches, int nxPatch,
      int nyPatch, int nfPatch)
{
   patches = KernelConn::createWeights(patches, nPatches, nxPatch, nyPatch, nfPatch);

   const int arbor = 0;
   int num_geisler_patches = numDataPatches(arbor);

   avePreActivity = (pvdata_t *) calloc(sizeof(pvdata_t), num_geisler_patches);
   avePostActivity = (pvdata_t *) calloc(sizeof(pvdata_t), num_geisler_patches);

   assert(geislerPatches == NULL);
   geislerPatches = (PVPatch**) calloc(sizeof(PVPatch*), num_geisler_patches);
   assert(geislerPatches != NULL);

   for (int geislerIndex = 0; geislerIndex < num_geisler_patches; geislerIndex++) {
      geislerPatches[geislerIndex] = pvpatch_inplace_new(nxPatch, nyPatch, nfPatch);
      assert(geislerPatches[geislerIndex] != NULL );
   }

   return patches;
}

int ODDConn::deleteWeights()
{
   const int arbor = 0;

   for (int k = 0; k < numDataPatches(arbor); k++) {
      pvpatch_inplace_delete(geislerPatches[k]);
   }
   free(geislerPatches);
   free(avePreActivity);
   free(avePostActivity);

   return KernelConn::deleteWeights();
}

PVPatch ** ODDConn::initializeDefaultWeights(PVPatch ** patches, int numPatches)
{
   // patches should already be initialized to zero
   return patches;
}


int ODDConn::updateState(float time, float dt)
{
   int axonID = 0;
   int num_kernels = this->numDataPatches(axonID);
   numUpdates++;

   int nPreEx = pre->clayer->numExtended;
   const pvdata_t * aPre = pre->getLayerData();
   for (int kfPre = 0; kfPre < num_kernels; kfPre++){
      pvdata_t avePre = 0.0;
      for (int kPre = 0; kPre < nPreEx; kPre+=num_kernels){
         avePre += aPre[kPre] > 0 ? aPre[kPre] : 0;
      }
      avePre *= num_kernels / (float) nPreEx ;
      avePreActivity[kfPre] += avePre;
   }

   int nPost = post->clayer->numNeurons;
   const pvdata_t * aPost = post->getLayerData();
   for (int kfPost = 0; kfPost < num_kernels; kfPost++){
      pvdata_t avePost = 0.0;
      for (int kPost = 0; kPost < nPost; kPost+=num_kernels) {
         PVLayerLoc loc = post->clayer->loc;
         int kPostEx = kIndexExtended(kPost, loc.nx, loc.ny, num_kernels, loc.nb);
            avePost += aPost[kPostEx];
      }
      avePost *= num_kernels / (float) nPost ;
      avePostActivity[kfPost] += avePost;
   }

   // initialize geislerPatches
   int numWeights = nxp * nyp * nfp;
   for (int iKernel = 0; iKernel < num_kernels; iKernel++){
      pvdata_t * kernelWeights = kernelPatches[iKernel]->data;
      pvdata_t * geislerWeights = geislerPatches[iKernel]->data;
      for (int iWeight = 0; iWeight < numWeights; iWeight++){
         kernelWeights[iWeight] = geislerWeights[iWeight];
         geislerWeights[iWeight] = 0.0;
      }
   }

   int status = updateWeights(axonID);

   // normalize geislerPatches
   for (int iKernel = 0; iKernel < num_kernels; iKernel++){
      pvdata_t * kernelWeights = kernelPatches[iKernel]->data;
      pvdata_t * geislerWeights = geislerPatches[iKernel]->data;
      for (int iWeight = 0; iWeight < numWeights; iWeight++){
         geislerWeights[iWeight] /= nPost;
         geislerWeights[iWeight] += kernelWeights[iWeight];
#ifdef APPLY_GEISLER_WEIGHTS
         int kfPost = iWeight % num_kernels;
         kernelWeights[iWeight] /= numUpdates;
         pvdata_t kernelNorm =
            (avePreActivity[iKernel] / numUpdates ) * (avePostActivity[kfPost] / numUpdates );
//         kernelWeights[iWeight] -= kernelNorm;
#else
         kernelWeights[iWeight] = 0.0;
#endif
      }
   }
#ifdef APPLY_GEISLER_WEIGHTS
   this->normalizeWeights(this->kernelPatches, this->numDataPatches(axonID));
#endif
   return status;
}


int ODDConn::updateWeights(int axonID)
{
   pvdata_t aPreThresh = 0.0f;
   pvdata_t aPostThresh = 0.0f;

   // this stride is in extended space for post-synaptic activity and
   // STDP decrement variable
   int postStrideY = post->clayer->loc.nf * (post->clayer->loc.nx
                   + post->clayer->loc.halo.lt + post->clayer->loc.halo.rt);

   int num_pre_extended = pre->clayer->numExtended;
   assert(num_pre_extended == numWeightPatches(axonID));

   const pvdata_t * preLayerData = pre->getLayerData();

   aPreThresh = 0.0f;
   aPostThresh = 0.0;
   int nKernels = numDataPatches(axonID);

   for (int kPre = 0; kPre < num_pre_extended; kPre++) {
      PVAxonalArbor * arbor = axonalArbor(kPre, axonID);

      float aPre = preLayerData[kPre];
      if (aPre <= aPreThresh) continue;

      PVPatch * wPatch = arbor->weights;
      size_t postOffset = arbor->offset;
      const pvdata_t * aPost = &post->getLayerData()[postOffset];

      int nk  = wPatch->nf * wPatch->nx; // one line in x at a time
      int ny  = wPatch->ny;
      int sy  = wPatch->sy;

      int kfPre = kPre % nKernels;
      PVPatch * gPatch = geislerPatches[kfPre];
      PVPatch * kPatch = kernelPatches[kfPre];

      pvdata_t * data_head = kPatch->data;
      pvdata_t * data_begin = wPatch->data;
      size_t data_offset = data_begin - data_head;
      pvdata_t * gWeights = &gPatch->data[data_offset];

      for (int y = 0; y < ny; y++) {
         for (int k = 0; k < nk; k++) {
             gWeights[k] += aPre  * ( ( aPost[k] > aPostThresh ) ? aPost[k] : 0.0 );
         }
          // advance pointers in y
         gWeights += sy;
         // postActivity and M are extended layer
         aPost += postStrideY;
     }
   }

   return 0;
}

int ODDConn::writeWeights(float time, bool last)
{
#ifdef APPLY_GEISLER_WEIGHTS
	// do nothing, kernels are already up to date
#else
	// copy geislerPatches to kernelPatches
   const int axonID = 0;
   const int num_kernels = numDataPatches(axonID);
   const int num_weights = nxp * nyp * nfp;
   for (int iKernel = 0; iKernel < num_kernels; iKernel++){
      pvdata_t * kernelWeights = kernelPatches[iKernel]->data;
      pvdata_t * geislerWeights = geislerPatches[iKernel]->data;
      for (int iWeight = 0; iWeight < num_weights; iWeight++){
         int kfPost = iWeight % num_kernels;
         pvdata_t kernelNorm =
            (avePreActivity[iKernel] / numUpdates ) * (avePostActivity[kfPost] / numUpdates );
         kernelWeights[iWeight] = geislerWeights[iWeight];
         kernelWeights[iWeight] /= numUpdates;
//         kernelWeights[iWeight] -= kernelNorm;
         kernelWeights[iWeight] /= fabs(kernelNorm + (kernelNorm==0));
      }
   }
   this->normalizeWeights(this->kernelPatches, this->numDataPatches(axonID));
#endif
   return KernelConn::writeWeights(time, last);
}



PVPatch ** ODDConn::normalizeWeights(PVPatch ** patches, int numPatches)
{
   int axonID = 0;
   int num_kernels = this->numDataPatches(axonID);

  for (int kPatch = 0; kPatch < num_kernels; kPatch++) {
      PVPatch * wp = kernelPatches[kPatch];
      pvdata_t * w = wp->data;
      int kfSelf = kPatch;
      int kxSelf = (nxp / 2);
      int kySelf = (nyp / 2);
      int kSelf = kIndex(kxSelf, kySelf, kfSelf, nxp, nyp, nfp);
      w[kSelf] = 0.0f;
   }
   return KernelConn::normalizeWeights(patches, numPatches);
}


} // namespace PV
