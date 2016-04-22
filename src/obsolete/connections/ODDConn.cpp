/*
 * ODDConn.cpp
 *
 *  Created on: ?
 *      Author: Kenyon
 */

#include "ODDConn.hpp"
#include "../io/io.h"
#include "../utils/conversions.h"
#include "../include/pv_types.h"
#include <assert.h>
#include <string.h>
#include <float.h>

namespace PV {

/*
ODDConn::ODDConn()
{
   printf("ODDConn::ODDConn: running default constructor\n");
   initialize_base();
}
*/

ODDConn::ODDConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
                         HyPerLayer * post, ChannelType channel)
{
   initialize_base();
   initialize(name, hc, pre, post, channel, NULL, NULL);
   // constructWeights(NULL); // HyPerConn::constructWeights moved back into HyPerConn::initialize
}
ODDConn::ODDConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
                         HyPerLayer * post, ChannelType channel, InitWeights *weightInit)
{
   initialize_base();
   initialize(name, hc, pre, post, channel, NULL, weightInit);
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
   initialize(name, hc, pre, post, channel, filename, NULL);
   // constructWeights(filename); // HyPerConn::constructWeights moved back into HyPerConn::initialize
}
ODDConn::ODDConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, ChannelType channel, const char * filename, InitWeights *weightInit)
{
   initialize_base();
   initialize(name, hc, pre, post, channel, filename, weightInit);
   // constructWeights(filename); // HyPerConn::constructWeights moved back into HyPerConn::initialize
}

ODDConn::~ODDConn() {
   deleteWeights();
}

int ODDConn::initialize_base()
{
   numUpdates = 0;
   avePreActivity = NULL;
   avePostActivity = NULL;
   ODDPatches = NULL;
   return PV_SUCCESS; // return KernelConn::initialize_base();
}

int ODDConn::createArbors() {
   KernelConn::createArbors();
   ODDPatches = (PVPatch***) calloc(numberOfAxonalArborLists(), sizeof(PVPatch**));
   assert(ODDPatches!=NULL);
   return PV_SUCCESS; //should we check if allocation was successful?
}


pvdata_t *  ODDConn::createWeights(PVPatch *** patches, int nPatches, int nxPatch,
      int nyPatch, int nfPatch, int axonId)
{
   pvdata_t * data_patches = KernelConn::createWeights(patches, nPatches, nxPatch, nyPatch, nfPatch, axonId);

   //const int arbor = 0;
   int num_geisler_patches = numDataPatches();

   avePreActivity = (pvdata_t *) calloc(sizeof(pvdata_t), num_geisler_patches);
   avePostActivity = (pvdata_t *) calloc(sizeof(pvdata_t), num_geisler_patches);

   assert(ODDPatches[axonId] == NULL);
   PVPatch** newGeislerPatches = (PVPatch**) calloc(sizeof(PVPatch*), num_geisler_patches);
   assert(newGeislerPatches != NULL);
   ODDPatches[axonId] = newGeislerPatches;

   for (int geislerIndex = 0; geislerIndex < num_geisler_patches; geislerIndex++) {
      ODDPatches[axonId][geislerIndex] = pvpatch_new(nxPatch, nyPatch); // pvpatch_inplace_new(nxPatch, nyPatch, nfPatch);
      assert(ODDPatches[axonId][geislerIndex] != NULL );
   }

   return data_patches;
}

int ODDConn::deleteWeights()
{
   //const int arbor = 0;

   for(int aid=0;aid<numberOfAxonalArborLists();aid++) {
      for (int k = 0; k < numDataPatches(); k++) {
         pvpatch_inplace_delete(ODDPatches[aid][k]);
      }
      free(ODDPatches[aid]);
   }
   free(ODDPatches);
   free(avePreActivity);
   free(avePostActivity);

   return 0; // KernelConn::deleteWeights(); // KernelConn destructor will call KernelConn::deleteWeights()
}

PVPatch ** ODDConn::initializeDefaultWeights(PVPatch ** patches, int numPatches)
{
   // patches should already be initialized to zero
   return patches;
}


int ODDConn::updateState(float time, float dt)
{
   //int axonID = 0;
   int num_kernels = this->numDataPatches();
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

   int status;
   for(int arborID=0;arborID<numberOfAxonalArborLists();arborID++) {
      // initialize geislerPatches
      int numWeights = nxp * nyp * nfp;
      for (int iKernel = 0; iKernel < num_kernels; iKernel++){
         pvdata_t * kernelWeights = getKernelPatch(arborID, iKernel)->data;
         pvdata_t * geislerWeights = ODDPatches[arborID][iKernel]->data;
         for (int iWeight = 0; iWeight < numWeights; iWeight++){
            kernelWeights[iWeight] = geislerWeights[iWeight];
            geislerWeights[iWeight] = 0.0;
         }
      }

      status = updateWeights(arborID);

      // normalize geislerPatches
      for (int iKernel = 0; iKernel < num_kernels; iKernel++){
         pvdata_t * kernelWeights = getKernelPatch(arborID, iKernel)->data;
         pvdata_t * geislerWeights = ODDPatches[arborID][iKernel]->data;
         for (int iWeight = 0; iWeight < numWeights; iWeight++){
            geislerWeights[iWeight] /= nPost;
            geislerWeights[iWeight] += kernelWeights[iWeight];
#ifdef APPLY_ODD_WEIGHTS
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
#ifdef APPLY_ODD_WEIGHTS
      this->normalizeWeights(this->kernelPatches, this->numDataPatches(axonID), arborID);
#endif
   }
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
   assert(num_pre_extended == numWeightPatches());

   const pvdata_t * preLayerData = pre->getLayerData();

   aPreThresh = 0.0f;
   aPostThresh = 0.0;
   int nKernels = numDataPatches();

   for (int kPre = 0; kPre < num_pre_extended; kPre++) {
      // PVAxonalArbor * arbor = axonalArbor(kPre, axonID);

      float aPre = preLayerData[kPre];
      if (aPre <= aPreThresh) continue;

      PVPatch * wPatch = getWeights(kPre,axonID);
      size_t postOffset = getAPostOffset(kPre, axonID);
      const pvdata_t * aPost = &post->getLayerData()[postOffset];

      int nk  = nfp * wPatch->nx; // one line in x at a time
      int ny  = wPatch->ny;
      int sy  = syp;

      int kfPre = kPre % nKernels;
      PVPatch * gPatch = ODDPatches[axonID][kfPre];
      PVPatch * kPatch = getKernelPatch(axonID, kfPre);

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
#ifdef APPLY_ODD_WEIGHTS
	// do nothing, kernels are already up to date
#else
	// copy geislerPatches to kernelPatches
   //const int axonID = 0;
   for(int arborId=0;arborId<numberOfAxonalArborLists();arborId++) {
      const int num_kernels = numDataPatches();
      const int num_weights = nxp * nyp * nfp;
      for (int iKernel = 0; iKernel < num_kernels; iKernel++){
         pvdata_t * kernelWeights = getKernelPatch(arborId, iKernel)->data;
         pvdata_t * ODDWeights = ODDPatches[arborId][iKernel]->data;
         for (int iWeight = 0; iWeight < num_weights; iWeight++){
            int kfPost = iWeight % num_kernels;
            pvdata_t kernelNorm =
               (avePreActivity[iKernel] / numUpdates ) * (avePostActivity[kfPost] / numUpdates );
            kernelWeights[iWeight] = ODDWeights[iWeight];
            kernelWeights[iWeight] /= numUpdates;
   //         kernelWeights[iWeight] -= kernelNorm;
            kernelWeights[iWeight] /= fabs(kernelNorm + (kernelNorm==0));
         }
      }
      if (this->normalize_flag){
         this->normalizeWeights(this->getKernelPatches(arborId), this->numDataPatches(), arborId);
      }
   }
#endif
   return KernelConn::writeWeights(time, last);
}



int ODDConn::normalizeWeights(PVPatch ** patches, int numPatches, int arborId)
{
   //int axonID = 0;
   int num_kernels = this->numDataPatches();

  for (int kPatch = 0; kPatch < num_kernels; kPatch++) {
      PVPatch * wp = getKernelPatch(arborId, kPatch);
      pvdata_t * w = wp->data;
      int kfSelf = kPatch;
      int kxSelf = (nxp / 2);
      int kySelf = (nyp / 2);
      int kSelf = kIndex(kxSelf, kySelf, kfSelf, nxp, nyp, nfp);
      w[kSelf] = 0.0f;
   }
   return KernelConn::normalizeWeights(patches, numPatches, arborId);
}


} // namespace PV
