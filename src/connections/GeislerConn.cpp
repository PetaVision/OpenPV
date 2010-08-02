/*
 * CocircConn.cpp
 *
 *  Created on: Nov 10, 2008
 *      Author: rasmussn
 */

#include "GeislerConn.hpp"
#include "../io/io.h"
#include "../utils/conversions.h"
#include "../include/pv_types.h"
#include <assert.h>
#include <string.h>
#include <float.h>

namespace PV {

GeislerConn::GeislerConn()
{
   printf("GeislerConn::GeislerConn: running default constructor\n");
   initialize_base();
}

GeislerConn::GeislerConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, int channel)
{
   initialize_base();
   initialize(name, hc, pre, post, channel);
}

GeislerConn::GeislerConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post)
{
   initialize_base();
   initialize(name, hc, pre, post, channel, NULL); // use default channel
}

// provide filename or set to NULL
GeislerConn::GeislerConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, int channel, const char * filename)
{
   initialize_base();
   initialize(name, hc, pre, post, channel, filename);
}

int GeislerConn::initialize_base()
{
   numUpdates = 0;
   avePreActivity = NULL;
   avePostActivity = NULL;
   geislerPatches = NULL;
   return KernelConn::initialize_base();
}


PVPatch ** GeislerConn::createWeights(PVPatch ** patches, int nPatches, int nxPatch,
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

int GeislerConn::deleteWeights()
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

PVPatch ** GeislerConn::initializeDefaultWeights(PVPatch ** patches, int numPatches)
{
   // patches should already be initialized to zero
   return patches;
}


int GeislerConn::updateState(float time, float dt)
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
         int kPostEx = kIndexExtended(kPost, loc.nx, loc.ny, num_kernels, loc.nPad);
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


int GeislerConn::updateWeights(int axonID)
{
   pvdata_t aPreThresh = 0.0f;
   pvdata_t aPostThresh = 0.0f;

   // this stride is in extended space for post-synaptic activity and
   // STDP decrement variable
   int postStrideY = post->clayer->numFeatures
   * (post->clayer->loc.nx + 2 * post->clayer->loc.nPad);

   int num_pre_extended = pre->clayer->numExtended;
   assert(num_pre_extended == numWeightPatches(axonID));

   const pvdata_t * preLayerData = pre->getLayerData();

   //TODO! following method is not MPI compatible (gives different answers depending on partition)
#define TRAINING_G1_TRIALS
#ifdef TRAINING_G1_TRIALS

  pvdata_t aPreMax = -FLT_MAX;
   for (int kPre = 0; kPre < num_pre_extended; kPre++) {
      aPreMax = (aPreMax > preLayerData[kPre]) ? aPreMax : preLayerData[kPre];
   }
   aPreThresh = aPreMax / 2.0f;
#else
   aPreThresh = 0.0f;
#endif

   const pvdata_t * postLayerData = post->getLayerData();

#ifdef TRAINING_G1_TRIALS
   pvdata_t aPostMax = -FLT_MAX;
   int num_post = post->clayer->numNeurons;
   for (int kPost = 0; kPost < num_post; kPost++) {
      int kPostEx = kIndexExtended(kPost,
                               post->clayer->loc.nx,
                               post->clayer->loc.ny,
                               post->clayer->loc.nBands,
                               post->clayer->loc.nPad);
      aPostMax = (aPostMax > postLayerData[kPostEx]) ? aPostMax : postLayerData[kPostEx];
   }
   aPostThresh = aPostMax / 2.0f;
#else
   aPostThresh = 0.0;
#endif

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
             gWeights[k] += aPre  * ( aPost[k] > aPostThresh ) ? aPost[k] : 0.0;
         }
          // advance pointers in y
         gWeights += sy;
         // postActivity and M are extended layer
         aPost += postStrideY;
     }
   }

   return 0;
}

int GeislerConn::writeWeights(float time, bool last)
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



PVPatch ** GeislerConn::normalizeWeights(PVPatch ** patches, int numPatches)
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
