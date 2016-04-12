/*
 * SparseConn.cpp
 */

#include "SparseConn.hpp"
#include "include/default_params.h"
#include "io/io.h"
#include "io/fileio.hpp"
#include "utils/conversions.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#include <iostream>
#include <cmath>
#include <utility>

#include "layers/accumulate_functions.h"
#include "weightinit/InitWeights.hpp"
#include "normalizers/NormalizeBase.hpp"
#include "privateTransposeConn.hpp"
#include "PlasticCloneConn.hpp"
#include "io/CoreParamGroupHandler.hpp"
#include <limits>

#include "utils/PVLog.hpp"
#include "utils/PVAlloc.hpp"
#include "utils/PVAssert.hpp"

namespace PV {

SparseConn::SparseConn()
{
   initialize_base();
}

SparseConn::SparseConn(const char * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer) {
   initialize_base();
   initialize(name, hc, weightInitializer, weightNormalizer);
}

SparseConn::~SparseConn() {
}

int SparseConn::initialize_base()
{
   return PV_SUCCESS;
}

int SparseConn::initialize(const char * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer) {
   super::initialize(name, hc, weightInitializer, weightNormalizer);
   _sparseWeightsAllocated.resize(numAxonalArborLists);
   std::fill(_sparseWeightsAllocated.begin(), _sparseWeightsAllocated.end(), false);
   return PV_SUCCESS;
}

void SparseConn::ioParam_sparsity(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "sparsity"));
   parent->ioParamValue(ioFlag, name, "sparsity", &_sparsity, 0.0f, true);
}

#if 0
void SparseConn::allocateSparseWeights(const char *logPrefix) {
   size_t totalWeights = numWeights();
   size_t patchSize = nfp * nxp * nyp;
   _sparseWeightInfo = findPercentileThreshold(0.0, wDataStart, numAxonalArborLists, numDataPatches, patchSize);
   size_t numSparse = _sparseWeightInfo.size;
   pvwdata_t threshold = _sparseWeightInfo.thresholdWeight;

   // Dealloc existing sparse data structs.
   _sparseWeight.clear();
   _sparsePost.clear();
   _patchSparseWeightCount.clear();
   _patchSparseWeightIndex.clear();

   // Reserve memory for sparse data structs
   _sparseWeight.reserve(numSparse);
   _sparsePost.reserve(numSparse);
   _patchSparseWeightIndex.reserve(getNumDataPatches());
   _patchSparseWeightCount.reserve(getNumDataPatches());

#if 0
   // Straight copy of patch data to sparse data structure, but no sparseness!
   for (int ar = 0; ar < numAxonalArborLists; ar++) {
      for (int pt = 0; pt < getNumDataPatches(); pt++) {
         for (int idx = 0; idx < nxp * nyp * nfp; idx++) {
            pvwdata_t weight = wDataStart[ar][pt * patchSize + idx];
            if (fabsf(weight) > 0) {
               Debug() << "Weight is above threshold: " << weight << std::endl;
               _sparseWeight.push_back(weight);
            }
         }
      }
   }

   Debug() << getName() << " " << _sparseWeight.size() << std::endl;
   return;
#endif

   int sy = getPostNonextStrides()->sy;

   for (int ar = 0; ar < numAxonalArborLists; ar++) {
      // Loop over all independent data patches, add the weights that are above the threshold
      for (int pt = 0; pt < getNumDataPatches(); pt++) {
         // Record the starting point for this patch in the sparse data table
         _patchSparseWeightIndex.push_back(_sparseWeight.size());

         // Get the start of the weight data for this patch
         pvwdata_t *weight = &wDataStart[ar][pt * patchSize];

         // Iterate over weights, add them to the sparse weight data list
         // if they're below the threshold
         int sparseWeightCount = 0;
         int pos = 0;

         // Add above threshold weights to the sparse data structure
         for (int y = 0; y < nyp; y++) {
            int postIdx = y * sy;
            for (int k = 0; k < nxp * nfp; k++, postIdx++, pos++) {
               pvwdata_t w = weight[pos];
               if (fabsf(w) >= threshold) {
                  _sparseWeight.push_back(w);
                  _sparsePost.push_back(postIdx);
                  sparseWeightCount++;
	       }
            }
         }

         _patchSparseWeightCount.push_back(sparseWeightCount);
      }
   }

   _sparseWeightsAllocated = true;

#if 1
   float sparsePercentage = (1.0f - float(_sparseWeight.size()) / float(totalWeights)) * 100.0f;
   float sparseSize = float(_sparseWeight.size() * sizeof(WeightListType::value_type)) / 1024.0 / 1024.0;

   Debug() << getName() << ":" << logPrefix << std::endl;
   Debug() << " Threshold:               " << threshold << std::endl;
   Debug() << " Num weight data patches: " << getNumDataPatches() << std::endl;
   Debug() << " Num weight patches:      " << getNumWeightPatches() << std::endl;
   Debug() << " Weight sparse ratio:     " << _sparseWeight.size() << "/" << totalWeights << "(" << sparsePercentage << "%" << sparseSize << " MB)" << std::endl;
   Debug() << " expected / actual:       " << _sparseWeightInfo.size << "/" << _sparseWeight.size() << std::endl;
#endif
}
#endif

/**
 * Find the weight value that that is in the nth percentile
 */
SparseWeightInfo findPercentileThreshold(float percentile, pvwdata_t **wDataStart, size_t numAxonalArborLists, size_t numPatches, size_t patchSize) {
   pvAssert(percentile >= 0.0f);
   pvAssert(percentile <= 1.0f);

   size_t fullWeightSize = numAxonalArborLists * numPatches * patchSize;
   SparseWeightInfo info;
   info.percentile = percentile;

   if (percentile >= 1.0) {
      info.size = fullWeightSize;
      info.thresholdWeight = 0.0;
      return info;
   }

   std::vector<pvwdata_t> weights;
   weights.reserve(fullWeightSize);

   for (int ar = 0; ar < numAxonalArborLists; ar++) {
      for (int pt = 0; pt < numPatches; pt++) {
         pvwdata_t *weight = &wDataStart[ar][pt * patchSize];
         for (int k = 0; k < patchSize; k++) {
            weights.push_back(fabs(weight[k]));
         }
      }
   }

   std::sort(weights.begin(), weights.end());
   int index = weights.size() * info.percentile;

   info.thresholdWeight = weights[index];
   info.size = weights.size() - index;
   return info;
}


void SparseConn::calculateSparseWeightInfo() {
   size_t patchSize = nfp * nxp * nyp;
   _sparseWeightInfo = findPercentileThreshold(_sparsity, get_wDataStart(), numAxonalArborLists, numDataPatches, patchSize);
}

void SparseConn::allocateSparseWeightsPre(PVLayerCube const *activity, int arbor) {
   calculateSparseWeightInfo();

   std::map<const WeightType * const, int> sizes;

   for (int kPreExt = 0; kPreExt < activity->numItems; kPreExt++) {
      PVPatch *patch = getWeights(kPreExt, arbor);
      const int nk = patch->nx * fPatchSize();
      const int nyp = patch->ny;
      const WeightType * const weightDataStart = get_wData(arbor, kPreExt);

      for (int y = 0; y < nyp; y++) {
         const WeightType * const weightPtr = weightDataStart + y * yPatchStride();

         // Don't re-sparsify something that's already been put thru the sparsfication grinder
         bool shouldSparsify = false;

         // Find the weight pointers for this nk sized patch
         WeightMapType::iterator sparseWeightValuesNk = _sparseWeightValues.find(nk);
         IndexMapType::iterator sparseWeightIndexesNk = _sparseWeightIndexes.find(nk);

         if (sparseWeightValuesNk == _sparseWeightValues.end()) {
            // Weight pointers don't exist for this sized nk. Allocate a map for this nk
            _sparseWeightValues.insert(make_pair(nk, WeightPtrMapType()));
            _sparseWeightIndexes.insert(make_pair(nk, WeightIndexMapType()));
            // Get references
            sparseWeightValuesNk = _sparseWeightValues.find(nk);
            sparseWeightIndexesNk = _sparseWeightIndexes.find(nk);
            shouldSparsify = true;
         } else if (sparseWeightValuesNk->second.find(weightPtr) == sparseWeightValuesNk->second.end()) {
            // This nk group exists, but no weight pointer.
            shouldSparsify = true;
         }

         if (shouldSparsify) {
            WeightListType sparseWeight;
            IndexListType idx;

            // Equivalent to inner loop accumulate
            for (int k = 0; k < nk; k++) {
               WeightType weight = weightPtr[k];
               if (std::abs(weight) >= _sparseWeightInfo.thresholdWeight) {
                  sparseWeight.push_back(weight);
                  idx.push_back(k);
               }
            }

            sparseWeightValuesNk->second.insert(make_pair(weightPtr, sparseWeight));
            sparseWeightIndexesNk->second.insert(make_pair(weightPtr, idx));
         }
      }

      _kPreExtWeightSparsified.insert(kPreExt);
   }

   _sparseWeightsAllocated[arbor] = true;
}

void SparseConn::allocateSparseWeightsPost(PVLayerCube const *activity, int arbor) {
   calculateSparseWeightInfo();
   const PVLayerLoc *targetLoc = post->getLayerLoc();
   const PVHalo *targetHalo = &targetLoc->halo;
   const int targetNx = targetLoc->nx;
   const int targetNy = targetLoc->ny;
   const int targetNf = targetLoc->nf;

   for (int kTargetRes = 0; kTargetRes < post->getNumNeurons(); kTargetRes++) {
      // Change restricted to extended post neuron
      int kTargetExt = kIndexExtended(kTargetRes, targetNx, targetNy, targetNf, targetHalo->lt, targetHalo->rt, targetHalo->dn, targetHalo->up);
      // get source layer's patch y stride
      int syp = postConn->yPatchStride();
      int yPatchSize = postConn->yPatchSize();
      // Iterate through y patch
      int nk = postConn->xPatchSize() * postConn->fPatchSize();
      int kernelIndex = postConn->patchToDataLUT(kTargetExt);

      const WeightType * const weightDataStart = postConn->get_wDataHead(arbor, kernelIndex);

      for (int ky = 0; ky < yPatchSize; ky++) {
         const WeightType * const weightPtr = weightDataStart + ky * syp;

         // Don't re-sparsify something that's already been put thru the sparsfication grinder
         bool shouldSparsify = false;

         // Find the weight pointers for this nk sized patch
         // Find the weight pointers for this nk sized patch
         WeightMapType::iterator sparseWeightValuesNk = _sparseWeightValues.find(nk);
         IndexMapType::iterator sparseWeightIndexesNk = _sparseWeightIndexes.find(nk);

         if (_sparseWeightValues.find(nk) == _sparseWeightValues.end()) {
            // Weight pointers don't exist for this sized nk. Allocate a map for this nk
            _sparseWeightValues.insert(make_pair(nk, WeightPtrMapType()));
            _sparseWeightIndexes.insert(make_pair(nk, WeightIndexMapType()));
            // Get references
            sparseWeightValuesNk = _sparseWeightValues.find(nk);
            sparseWeightIndexesNk = _sparseWeightIndexes.find(nk);
            shouldSparsify = true;
         } else if (sparseWeightValuesNk->second.find(weightPtr) == sparseWeightValuesNk->second.end()) {
            // This nk group exists, but no weight pointer.
            shouldSparsify = true;
         }

         if (shouldSparsify) {
            WeightListType sparseWeight;
            IndexListType idx;

            for (int k = 0; k < nk; k++) {
               WeightType weight = weightPtr[k];
               if (std::abs(weight) >= _sparseWeightInfo.thresholdWeight) {
                  sparseWeight.push_back(weight);
                  idx.push_back(k);
               }
            }

            sparseWeightValuesNk->second.insert(make_pair(weightPtr, sparseWeight));
            sparseWeightIndexesNk->second.insert(make_pair(weightPtr, idx));
         }
      }

      _kPreExtWeightSparsified.insert(kTargetRes);
   }
   
   _sparseWeightsAllocated[arbor] = true;
}

void SparseConn::deliverOnePreNeuronActivity(int kPreExt, int arbor, pvadata_t a, pvgsyndata_t * postBufferStart, void * auxPtr) {
   pvAssert(_sparseWeightsAllocated[arbor] == true);
   pvAssert(_kPreExtWeightSparsified.find(kPreExt) != _kPreExtWeightSparsified.end());

   PVPatch *patch = getWeights(kPreExt, arbor);
   const int nk = patch->nx * fPatchSize();
   const int nyp = patch->ny;
   const int sy  = getPostNonextStrides()->sy;       // stride in layer
   WeightType *weightDataStart = get_wData(arbor, kPreExt);
   pvgsyndata_t *postPatchStart = postBufferStart + getGSynPatchStart(kPreExt, arbor);
   int offset = 0;
   int sf = 1;

   for (int y = 0; y < nyp; y++) {
      WeightType *weightPtr = weightDataStart + y * yPatchStride();
      pvadata_t *post = postPatchStart + y * sy + offset;

      const WeightListType& sparseWeights = _sparseWeightValues.find(nk)->second.find(weightPtr)->second;
      const IndexListType& idx = _sparseWeightIndexes.find(nk)->second.find(weightPtr)->second;

      for (int k = 0; k < sparseWeights.size(); k++) {
         int outIdx = idx[k];
         post[outIdx] += a * sparseWeights[k];
      }
   }
}

void SparseConn::deliverOnePostNeuronActivity(int arborID, int kTargetExt, int inSy, float* activityStartBuf, pvdata_t* gSynPatchPos, float dt_factor, taus_uint4 * rngPtr) {
   // get source layer's patch y stride
   int syp = postConn->yPatchStride();
   int yPatchSize = postConn->yPatchSize();
   // Iterate through y patch
   int nk = postConn->xPatchSize() * postConn->fPatchSize();
   int kernelIndex = postConn->patchToDataLUT(kTargetExt);

   pvwdata_t* weightStartBuf = postConn->get_wDataHead(arborID, kernelIndex);
   int offset = 0;
   for (int ky = 0; ky < yPatchSize; ky++) {
      float * activityY = &(activityStartBuf[ky*inSy+offset]);
      pvwdata_t *weightPtr = weightStartBuf + ky * syp;

      const WeightListType& sparseWeight = _sparseWeightValues.find(nk)->second.find(weightPtr)->second;
      const IndexListType& idx = _sparseWeightIndexes.find(nk)->second.find(weightPtr)->second;

      float dv = 0.0;
      for (int k = 0; k < sparseWeight.size(); k++) {
         dv += activityY[idx[k]] * sparseWeight[k];
      }
      *gSynPatchPos += dt_factor * dv;
   }
}


int SparseConn::deliverPresynapticPerspective(PVLayerCube const * activity, int arborID) {
   if (!_sparseWeightsAllocated[arborID]) {
      allocateSparseWeightsPre(activity, arborID);
   }

   //Check if we need to update based on connection's channel
   if (getChannel() == CHANNEL_NOUPDATE) {
      return PV_SUCCESS;
   }
   pvAssert(post->getChannel(getChannel()));

   float dt_factor = getConvertToRateDeltaTimeFactor();
   if (getPvpatchAccumulateType() == ACCUMULATE_STOCHASTIC) {
      dt_factor = getParent()->getDeltaTime();
   }

   const PVLayerLoc * preLoc = preSynapticLayer()->getLayerLoc();
   const PVLayerLoc * postLoc = postSynapticLayer()->getLayerLoc();

   pvAssert(arborID >= 0);

   const int numExtended = activity->numItems;

   int nbatch = parent->getNBatch();

   for (int b = 0; b < nbatch; b++) {
      const int offset = b * (preLoc->nx + preLoc->halo.rt + preLoc->halo.lt) * (preLoc->ny + preLoc->halo.up + preLoc->halo.dn) * preLoc->nf;
      pvdata_t * activityBatch = activity->data + offset;
      pvdata_t * gSynPatchHeadBatch = post->getChannel(getChannel()) + b * postLoc->nx * postLoc->ny * postLoc->nf;
      unsigned int * activeIndicesBatch = NULL;
      int numLoop = numExtended;

      if (activity->isSparse) {
         activeIndicesBatch = activity->activeIndices + offset;
         numLoop = activity->numActive[b];
      }

#ifdef PV_USE_OPENMP_THREADS
      // Clear all thread gsyn buffer
      if (thread_gSyn) {
         int numNeurons = post->getNumNeurons();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int i = 0; i < parent->getNumThreads() * numNeurons; i++) {
            int ti = i/numNeurons;
            int ni = i % numNeurons;
            thread_gSyn[ti][ni] = 0;
         }
      }
#endif // PV_USE_OPENMP_THREADS

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
      for (int loopIndex = 0; loopIndex < numLoop; loopIndex++) {
         int kPreExt = loopIndex;
         if (activity->isSparse) {
            kPreExt = activeIndicesBatch[loopIndex];
         }

         float a = activityBatch[kPreExt] * dt_factor;
         if (a == 0.0f) continue;

         // If we're using thread_gSyn, set this here
         pvdata_t *gSynPatchHead = gSynPatchHeadBatch;
#ifdef PV_USE_OPENMP_THREADS
         if (thread_gSyn) {
            int ti = omp_get_thread_num();
            gSynPatchHead = thread_gSyn[ti];
         }
#else // PV_USE_OPENMP_THREADS
         gSynPatchHead = gSynPatchHeadBatch;
#endif // PV_USE_OPENMP_THREADS
         deliverOnePreNeuronActivity(kPreExt, arborID, a, gSynPatchHead, getRandState(kPreExt));
      }
#ifdef PV_USE_OPENMP_THREADS
      // Accumulate back into gSyn // Should this be done in HyPerLayer where it can be done once, as opposed to once per connection?
      if (thread_gSyn) {
         pvdata_t * gSynPatchHead = gSynPatchHeadBatch;
         int numNeurons = post->getNumNeurons();
         //Looping over neurons first to be thread safe
#pragma omp parallel for
         for (int ni = 0; ni < numNeurons; ni++) {
            for(int ti = 0; ti < parent->getNumThreads(); ti++) {
               gSynPatchHead[ni] += thread_gSyn[ti][ni];
            }
         }
      }
#endif
   }
   return PV_SUCCESS;
}

int SparseConn::deliverPostsynapticPerspective(PVLayerCube const * activity, int arborID) {
   if (!_sparseWeightsAllocated[arborID]) {
      allocateSparseWeightsPost(activity, arborID);
   }

   // Check channel number for noupdate
   if(getChannel() == CHANNEL_NOUPDATE){
      return PV_SUCCESS;
   }
   pvAssert(post->getChannel(getChannel()));
   pvAssert(arborID >= 0);

   //Get number of neurons restricted target
   const int numPostRestricted = post->getNumNeurons();

   float dt_factor = getConvertToRateDeltaTimeFactor();
   if (getPvpatchAccumulateType() == ACCUMULATE_STOCHASTIC) {
      dt_factor = getParent()->getDeltaTime();
   }

   const PVLayerLoc * sourceLoc = preSynapticLayer()->getLayerLoc();
   const PVLayerLoc * targetLoc = post->getLayerLoc();

   const int sourceNx = sourceLoc->nx;
   const int sourceNy = sourceLoc->ny;
   const int sourceNf = sourceLoc->nf;
   const int targetNx = targetLoc->nx;
   const int targetNy = targetLoc->ny;
   const int targetNf = targetLoc->nf;
   const int nbatch = targetLoc->nbatch;

   const PVHalo * sourceHalo = &sourceLoc->halo;
   const PVHalo * targetHalo = &targetLoc->halo;

   //get source layer's extended y stride
   int sy  = (sourceNx+sourceHalo->lt+sourceHalo->rt)*sourceNf;

   //The start of the gsyn buffer
   pvdata_t * gSynPatchHead = post->getChannel(getChannel());

   long * startSourceExtBuf = getPostToPreActivity();
   if (!startSourceExtBuf) {
      exitFailure("HyPerLayer::recvFromPost error getting preToPostActivity from connection. Is shrink_patches on?");
   }

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static) collapse(2)
#endif
   for(int b = 0; b < nbatch; b++) {
      for (int kTargetRes = 0; kTargetRes < numPostRestricted; kTargetRes++) {
         pvdata_t * activityBatch = activity->data + b * (sourceNx + sourceHalo->rt + sourceHalo->lt) * (sourceNy + sourceHalo->up + sourceHalo->dn) * sourceNf;
         pvdata_t * gSynPatchHeadBatch = gSynPatchHead + b * targetNx * targetNy * targetNf;
         //Change restricted to extended post neuron
         int kTargetExt = kIndexExtended(kTargetRes, targetNx, targetNy, targetNf, targetHalo->lt, targetHalo->rt, targetHalo->dn, targetHalo->up);

         //Read from buffer
         long startSourceExt = startSourceExtBuf[kTargetRes];

         //Calculate target's start of gsyn
         pvdata_t * gSynPatchPos = gSynPatchHeadBatch + kTargetRes;

         taus_uint4 * rngPtr = getRandState(kTargetRes);
         float* activityStartBuf = &(activityBatch[startSourceExt]); 

         deliverOnePostNeuronActivity(arborID, kTargetExt, sy, activityStartBuf, gSynPatchPos, dt_factor, rngPtr);
      }
   }
   return PV_SUCCESS;
}

} // namespace PV
