/*
 * CocircConn.cpp
 *
 *  Created on: Nov 10, 2008
 *      Author: rasmussn
 */

#include "GeislerConn.hpp"
#include "../PetaVision/src/io/io.h"
#include "../PetaVision/src/utils/conversions.h"
#include "../PetaVision/src/include/pv_types.h"
#include <assert.h>
#include <string.h>

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
	avePreActivity = 0.0;
	avePostActivity = 0.0;
	geislerPatches = NULL;
	return KernelConn::initialize_base();
}


PVPatch ** GeislerConn::createWeights(PVPatch ** patches, int nPatches, int nxPatch,
      int nyPatch, int nfPatch)
{
	const int arbor = 0;
	int numGeislerPatches = numDataPatches(arbor);

	geislerPatches = (PVPatch**) calloc(sizeof(PVPatch*), numGeislerPatches);
	assert(geislerPatches != NULL);

	for (int geislerIndex = 0; geislerIndex < numGeislerPatches; geislerIndex++) {
		geislerPatches[geislerIndex] = pvpatch_inplace_new(nxPatch, nyPatch, nfPatch);
		assert(geislerPatches[geislerIndex] != NULL );
	}

	return KernelConn::createWeights(patches, nPatches, nxPatch, nyPatch, nfPatch);
}

int GeislerConn::deleteWeights()
{
   const int arbor = 0;

   for (int k = 0; k < numDataPatches(arbor); k++) {
      pvpatch_inplace_delete(geislerPatches[k]);
   }
   free(geislerPatches);

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
	numUpdates++;
	int nPreEx = pre->clayer->numExtended;
	const pvdata_t * aPre = pre->getLayerData();
	pvdata_t avePre = 0.0;
	for (int kPre = 0; kPre < nPreEx; kPre++){
		avePre += aPre[kPre];
	}
	avePre /= nPreEx;
	avePreActivity += avePre;
	int nPost = post->clayer->numNeurons;
	const pvdata_t * aPost = post->getLayerData();

	pvdata_t avePost = 0.0;
	for (int kPost = 0; kPost < nPost; kPost++){
		avePost += aPost[kPost];
	}
	avePost /= nPost;
	avePostActivity += avePost;

	// initialize geislerPatches
	int numKernels = this->numDataPatches(axonID);
	int numWeights = nxp * nyp * nfp;
	for (int iKernel = 0; iKernel < numKernels; iKernel++){
		pvdata_t * kernelWeights = kernelPatches[iKernel]->data;
		pvdata_t * geislerWeights = geislerPatches[iKernel]->data;
		for (int iWeight = 0; iWeight < numWeights; iWeight++){
			kernelWeights[iWeight] = geislerWeights[iWeight];
			geislerWeights[iWeight] = 0.0;
		}
	}

	int status = updateWeights(axonID);

	// normalize geislerPatches
	for (int iKernel = 0; iKernel < numKernels; iKernel++){
		pvdata_t * kernelWeights = kernelPatches[iKernel]->data;
		pvdata_t * geislerWeights = geislerPatches[iKernel]->data;
		for (int iWeight = 0; iWeight < numWeights; iWeight++){
			geislerWeights[iWeight] /= nPost;
			geislerWeights[iWeight] += kernelWeights[iWeight];
#ifdef APPLY_GEISLER_WEIGHTS
			kernelWeights[iWeight] /= numUpdates;
			pvdata_t kernelNorm = (avePreActivity / numUpdates ) * (avePostActivity / numUpdates );
			kernelWeights[iWeight] -= kernelNorm;
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
	// this stride is in extended space for post-synaptic activity and
	// STDP decrement variable
	int postStrideY = post->clayer->numFeatures
                         * (post->clayer->loc.nx + 2 * post->clayer->loc.nPad);

	int numExtended = pre->clayer->numExtended;
	assert(numExtended == numWeightPatches(axonID));

	const pvdata_t * preLayerData = pre->getLayerData();

	int nKernels = numDataPatches(axonID);

	for (int kPre = 0; kPre < numExtended; kPre++) {
      PVAxonalArbor * arbor = axonalArbor(kPre, axonID);

      float aPre = preLayerData[kPre];
      if (aPre == 0.0) continue;

      PVPatch * wPatch = arbor->weights;
      size_t postOffset = arbor->offset;
      const pvdata_t * aPost = &post->getLayerData()[postOffset];

      int nk  = wPatch->nf * wPatch->nx; // one line in x at a time
      int ny  = wPatch->ny;
      int sy  = wPatch->sy;

      int kfPre = kPre % nKernels;
      PVPatch * gPatch = geislerPatches[kfPre];
      PVPatch * kPatch = kernelPatches[kfPre];

      pvdata_t * data_head = (pvdata_t *) ((char*) kPatch + sizeof(PVPatch));
      pvdata_t * data_begin = kPatch->data;
      size_t data_offset = data_begin - data_head;
      pvdata_t * gWeights = &gPatch->data[data_offset];

      for (int y = 0; y < ny; y++) {
         for (int k = 0; k < nk; k++) {
             gWeights[k] += aPre  * aPost[k];
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
	const int numKernels = numDataPatches(axonID);
	const int numWeights = nxp * nyp * nfp;
	for (int iKernel = 0; iKernel < numKernels; iKernel++){
			pvdata_t * kernelWeights = kernelPatches[iKernel]->data;
			pvdata_t * geislerWeights = geislerPatches[iKernel]->data;
			for (int iWeight = 0; iWeight < numWeights; iWeight++){
				kernelWeights[iWeight] = geislerWeights[iWeight];
				kernelWeights[iWeight] /= numUpdates;
				pvdata_t kernelNorm = (avePreActivity / numUpdates ) * (avePostActivity / numUpdates );
				kernelWeights[iWeight] -= kernelNorm;
			}
		}
	this->normalizeWeights(this->kernelPatches, this->numDataPatches(axonID));
#endif
	return KernelConn::writeWeights(time, last);
}



} // namespace PV
