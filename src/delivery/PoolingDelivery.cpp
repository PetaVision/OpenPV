/*
 * PoolingDelivery.cpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#include "PoolingDelivery.hpp"
#include "columns/HyPerCol.hpp"
#include "delivery/accumulate_functions.hpp"
#include "utils/MapLookupByType.hpp"

namespace PV {

PoolingDelivery::PoolingDelivery(char const *name, HyPerCol *hc) { initialize(name, hc); }

PoolingDelivery::PoolingDelivery() {}

PoolingDelivery::~PoolingDelivery() {}

int PoolingDelivery::initialize(char const *name, HyPerCol *hc) {
   return BaseDelivery::initialize(name, hc);
}

int PoolingDelivery::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseDelivery::ioParamsFillGroup(ioFlag);
   ioParam_pvpatchAccumulateType(ioFlag);
   ioParam_updateGSynFromPostPerspective(ioFlag);
   ioParam_needPostIndexLayer(ioFlag);
   ioParam_postIndexLayerName(ioFlag);
   return PV_SUCCESS;
}

void PoolingDelivery::ioParam_pvpatchAccumulateType(enum ParamsIOFlag ioFlag) {
   PVParams *params = parent->parameters();

   parent->parameters()->ioParamStringRequired(
         ioFlag, name, "pvpatchAccumulateType", &mPvpatchAccumulateTypeString);
   if (ioFlag == PARAMS_IO_READ) {
      mAccumulateType = parseAccumulateTypeString(mPvpatchAccumulateTypeString);
      FatalIf(
            mAccumulateType == UNDEFINED,
            "pvpatchAccumulateType \"%s\" is unrecognized.\n"
            "  Allowed values are \"maxpooling\", \"sumpooling\", or \"avgpooling\".\n",
            mPvpatchAccumulateTypeString);
   }
}

PoolingDelivery::AccumulateType
PoolingDelivery::parseAccumulateTypeString(char const *poolingTypeString) {
   if (poolingTypeString == nullptr) {
      return UNDEFINED;
   }
   PoolingDelivery::AccumulateType accType;
   std::string str(poolingTypeString);
   // Convert string to lowercase so that capitalization doesn't matter.
   for (auto &c : str) {
      c = std::tolower(c, std::locale());
   }
   // "max_pooling", "max pooling", "maxpooling" are equally acceptable (same for
   // sum and avg)
   if (str.size() >= 4 && (str[3] == ' ' || str[3] == '_')) {
      str.erase(3, 1);
   }

   if (strcmp(str.c_str(), "maxpooling") == 0) {
      accType = MAXPOOLING;
   }
   else if (strcmp(str.c_str(), "sumpooling") == 0) {
      accType = SUMPOOLING;
   }
   else if (strcmp(str.c_str(), "avgpooling") == 0) {
      accType = AVGPOOLING;
   }
   else {
      accType = UNDEFINED;
   }
   return accType;
}

void PoolingDelivery::ioParam_updateGSynFromPostPerspective(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "receiveGpu"));
   if (!mReceiveGpu) {
      parent->parameters()->ioParamValue(
            ioFlag,
            name,
            "updateGSynFromPostPerspective",
            &mUpdateGSynFromPostPerspective,
            mUpdateGSynFromPostPerspective);
   }
}

void PoolingDelivery::ioParam_needPostIndexLayer(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(
         ioFlag, name, "needPostIndexLayer", &mNeedPostIndexLayer, mNeedPostIndexLayer);
}

void PoolingDelivery::ioParam_postIndexLayerName(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "needPostIndexLayer"));
   if (mNeedPostIndexLayer) {
      parent->parameters()->ioParamStringRequired(
            ioFlag, name, "postIndexLayerName", &mPostIndexLayerName);
   }
}

int PoolingDelivery::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status = BaseDelivery::communicateInitInfo(message);
   if (status != PV_SUCCESS) {
      return status;
   }

   mPatchSize = mapLookupByType<PatchSize>(message->mHierarchy, getDescription());
   pvAssert(mPatchSize);
   if (!mPatchSize->getInitInfoCommunicatedFlag()) {
      return PV_POSTPONE;
   }

   mWeightsPair = mapLookupByType<ImpliedWeightsPair>(message->mHierarchy, getDescription());
   pvAssert(mWeightsPair);
   if (!mWeightsPair->getInitInfoCommunicatedFlag()) {
      return PV_POSTPONE;
   }

   if (mUpdateGSynFromPostPerspective) {
      mWeightsPair->needPost();
   }
   else {
      mWeightsPair->needPre();
   }
   return PV_SUCCESS;
}

int PoolingDelivery::allocateDataStructures() {
   int status = BaseDelivery::allocateDataStructures();
   allocateThreadGSyn();
   return status;
}

void PoolingDelivery::allocateThreadGSyn() {
   // If multithreaded, allocate a GSyn buffer for each thread, to avoid collisions.
   int const numThreads = parent->getNumThreads();
   if (numThreads > 1) {
      mThreadGSyn.resize(numThreads);
      // mThreadGSyn is only a buffer for one batch element. We're threading over presynaptic
      // neuron index, not batch element; so batch elements will be processed serially.
      for (auto &th : mThreadGSyn) {
         th.resize(mPostLayer->getNumNeurons());
      }
   }
}

void PoolingDelivery::deliver() {
   // Check if we need to update based on connection's channel
   if (getChannelCode() == CHANNEL_NOUPDATE) {
      return;
   }

   if (mReceiveGpu) {
#ifdef PV_USE_CUDA
      deliverGPU();
#endif // PV_USE_CUDA
   }
   else {
      if (mUpdateGSynFromPostPerspective) {
         deliverPostsynapticPerspective();
      }
      else {
         deliverPresynapticPerspective();
      }
   }
}

void PoolingDelivery::deliverPostsynapticPerspective() {
   PVLayerLoc const *sourceLoc = mPreLayer->getLayerLoc();
   PVLayerLoc const *targetLoc = mPostLayer->getLayerLoc();
   Weights *postWeights        = mWeightsPair->getPostWeights();

   // Slightly inefficient to define the function pointer each time deliver() is called;
   // but the real inefficiency is calling the function pointer in a tight for-loop.
   // TODO: Use templating instead of function pointer.
   void (*accumulateFunctionPointer)(
         int kPreRes, int nk, float *v, float *a, float *w, void *auxPtr, int sf) = nullptr;
   switch (mAccumulateType) {
      case MAXPOOLING: accumulateFunctionPointer = pvpatch_max_pooling_from_post; break;
      case SUMPOOLING: accumulateFunctionPointer = pvpatch_sum_pooling_from_post; break;
      case AVGPOOLING:
         accumulateFunctionPointer = pvpatch_sum_pooling_from_post;
         // Division by the number of weights happens outside the call to the accumulate function.
         break;
      default:
         pvAssert(0);
         // Only MAXPOOLING, SUMPOOLING, AVGPOOLING are allowed.
         // UNDEFINED is the only possible value of mAccumulateType, but the type should be
         // defined before this function is ever called.
         break;
   }

   float w = 1.0f;
   if (mAccumulateType == AVGPOOLING) {
      float relative_XScale = pow(2, (getPostLayer()->getXScale() - getPreLayer()->getXScale()));
      float relative_YScale = pow(2, (getPostLayer()->getYScale() - getPreLayer()->getYScale()));
      float nxp             = (float)mPatchSize->getPatchSizeX();
      float nyp             = (float)mPatchSize->getPatchSizeY();
      w                     = 1.0f / (nxp * relative_XScale * nyp * relative_YScale);
   }

   int const numAxonalArbors = mArborList->getNumAxonalArbors();
   for (int arbor = 0; arbor < numAxonalArbors; arbor++) {
      int delay                = mArborList->getDelay(arbor);
      PVLayerCube activityCube = mPreLayer->getPublisher()->createCube(delay);

      float *gSyn = getPostLayer()->getChannel(getChannelCode());
      pvAssert(gSyn);

      // Get number of neurons restricted target
      int const numPostRestricted = mPostLayer->getNumNeurons();

      int const sourceNx = sourceLoc->nx;
      int const sourceNy = sourceLoc->ny;
      int const sourceNf = sourceLoc->nf;
      int const targetNx = targetLoc->nx;
      int const targetNy = targetLoc->ny;
      int const targetNf = targetLoc->nf;

      const PVHalo *sourceHalo = &sourceLoc->halo;
      const PVHalo *targetHalo = &targetLoc->halo;

      // get source layer's extended y stride
      int sy = (sourceNx + sourceHalo->lt + sourceHalo->rt) * sourceNf;

      clearGateIdxBuffer();
      float *gatePatchHead = nullptr;
      if (mNeedPostIndexLayer) {
         gatePatchHead = mPostIndexLayer->getChannel(CHANNEL_EXC);
      }

      float resetVal = 0.0f;
      if (mAccumulateType == MAXPOOLING) {
         resetVal = -INFINITY;
      }

      for (int b = 0; b < parent->getNBatch(); b++) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int kTargetRes = 0; kTargetRes < numPostRestricted; kTargetRes++) {
            float *activityBatch = activityCube.data
                                   + b * (sourceNx + sourceHalo->rt + sourceHalo->lt)
                                           * (sourceNy + sourceHalo->up + sourceHalo->dn)
                                           * sourceNf;
            float *gSynBatchHead = gSyn + b * targetNx * targetNy * targetNf;

            // Change restricted to extended post neuron
            int kTargetExt = kIndexExtended(
                  kTargetRes,
                  targetNx,
                  targetNy,
                  targetNf,
                  targetHalo->lt,
                  targetHalo->rt,
                  targetHalo->dn,
                  targetHalo->up);
            long startSourceExt = postWeights->getGeometry()->getUnshrunkenStart(kTargetExt);

            // Calculate target's start of gsyn
            float *gSynPatchPos = gSynBatchHead + kTargetRes;
            // Initialize patch as a huge negative number
            *gSynPatchPos = resetVal;

            float *gatePatchPos = nullptr;
            if (mNeedPostIndexLayer) {
               gatePatchPos = gatePatchHead + b * mPostIndexLayer->getNumNeurons() + kTargetRes;
               // Initialize gatePatchPos as a negative number
               *gatePatchPos = (float)-1;
            }

            float *activityStartBuf = &(activityBatch[startSourceExt]);

            int sf           = postWeights->getPatchSizeF();
            int yPatchSize   = postWeights->getPatchSizeY();
            int numPerStride = postWeights->getPatchSizeX() * postWeights->getPatchSizeF();

            const PVLayerLoc *postLoc = mPostLayer->getLayerLoc();
            int const kfPost          = featureIndex(
                  kTargetExt,
                  postLoc->nx + postLoc->halo.lt + postLoc->halo.rt,
                  postLoc->ny + postLoc->halo.dn + postLoc->halo.up,
                  postLoc->nf);
            int offset = kfPost;

            for (int ky = 0; ky < yPatchSize; ky++) {
               int kPreExt = startSourceExt + ky * sy + offset;
               int const kxPreExt =
                     kxPos(kPreExt,
                           sourceLoc->nx + sourceLoc->halo.lt + sourceLoc->halo.rt,
                           sourceLoc->ny + sourceLoc->halo.dn + sourceLoc->halo.up,
                           sourceLoc->nf);
               int const kyPreExt =
                     kyPos(kPreExt,
                           sourceLoc->nx + sourceLoc->halo.lt + sourceLoc->halo.rt,
                           sourceLoc->ny + sourceLoc->halo.dn + sourceLoc->halo.up,
                           sourceLoc->nf);
               int const kfPre = featureIndex(
                     kPreExt,
                     sourceLoc->nx + sourceLoc->halo.lt + sourceLoc->halo.rt,
                     sourceLoc->ny + sourceLoc->halo.dn + sourceLoc->halo.up,
                     sourceLoc->nf);
               int const kxPreGlobalExt = kxPreExt + sourceLoc->kx0;
               int const kyPreGlobalExt = kyPreExt + sourceLoc->ky0;
               int const kPreGlobalExt  = kIndex(
                     kxPreGlobalExt,
                     kyPreGlobalExt,
                     kfPre,
                     sourceLoc->nxGlobal + sourceLoc->halo.lt + sourceLoc->halo.rt,
                     sourceLoc->nyGlobal + sourceLoc->halo.up + sourceLoc->halo.dn,
                     sourceLoc->nf);

               float *activityY = &(activityStartBuf[ky * sy + offset]);

               (accumulateFunctionPointer)(
                     kPreGlobalExt, numPerStride, gSynPatchPos, activityY, &w, gatePatchPos, sf);
            }
         }
      }
   }
}

void PoolingDelivery::deliverPresynapticPerspective() {
   PVLayerLoc const *preLoc  = getPreLayer()->getLayerLoc();
   PVLayerLoc const *postLoc = getPostLayer()->getLayerLoc();
   Weights *preWeights       = mWeightsPair->getPreWeights();

   // Slightly inefficient to define the function pointer each time deliver() is called;
   // but the real inefficiency is calling the function pointer in a tight for-loop.
   // TODO: Use templating instead of function pointer.
   void (*accumulateFunctionPointer)(
         int kPreRes, int nk, float *v, float a, float *w, void *auxPtr, int sf) = nullptr;
   switch (mAccumulateType) {
      case MAXPOOLING: accumulateFunctionPointer = pvpatch_max_pooling; break;
      case SUMPOOLING: accumulateFunctionPointer = pvpatch_sum_pooling; break;
      case AVGPOOLING:
         accumulateFunctionPointer = pvpatch_sum_pooling;
         // Division by the number of weights happens outside the call to the accumulate function.
         break;
      default:
         pvAssert(0);
         // Only MAXPOOLING, SUMPOOLING, AVGPOOLING are allowed.
         // UNDEFINED is the only possible value of mAccumulateType, but the type should be
         // defined before this function is ever called.
         break;
   }

   float w = 1.0f;
   if (mAccumulateType == AVGPOOLING) {
      float relative_XScale = pow(2, (getPostLayer()->getXScale() - getPreLayer()->getXScale()));
      float relative_YScale = pow(2, (getPostLayer()->getYScale() - getPreLayer()->getYScale()));
      float nxp             = (float)mPatchSize->getPatchSizeX();
      float nyp             = (float)mPatchSize->getPatchSizeY();
      w                     = 1.0f / (nxp * relative_XScale * nyp * relative_YScale);
   }

   int const numAxonalArbors = mArborList->getNumAxonalArbors();
   for (int arbor = 0; arbor < numAxonalArbors; arbor++) {
      int delay                = mArborList->getDelay(arbor);
      PVLayerCube activityCube = mPreLayer->getPublisher()->createCube(delay);

      float *gSyn = getPostLayer()->getChannel(getChannelCode());
      pvAssert(gSyn);

      float resetVal = 0;
      if (mAccumulateType == MAXPOOLING) {
         resetVal = -INFINITY;
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int i = 0; i < getPostLayer()->getNumNeuronsAllBatches(); i++) {
            gSyn[i] = resetVal;
         }
      }

      clearGateIdxBuffer();

      for (int b = 0; b < mPreLayer->getLayerLoc()->nbatch; b++) {
         float *activityBatch = activityCube.data
                                + b * (preLoc->nx + preLoc->halo.rt + preLoc->halo.lt)
                                        * (preLoc->ny + preLoc->halo.up + preLoc->halo.dn)
                                        * preLoc->nf;
         float *gSynPatchHeadBatch = gSyn + b * postLoc->nx * postLoc->ny * postLoc->nf;
         float *gatePatchHeadBatch = NULL;
         if (mNeedPostIndexLayer) {
            gatePatchHeadBatch =
                  mPostIndexLayer->getChannel(CHANNEL_EXC) + b * mPostIndexLayer->getNumNeurons();
         }

         SparseList<float>::Entry const *activeIndicesBatch = NULL;
         if (activityCube.isSparse) {
            activeIndicesBatch = (SparseList<float>::Entry *)activityCube.activeIndices
                                 + b * (preLoc->nx + preLoc->halo.rt + preLoc->halo.lt)
                                         * (preLoc->ny + preLoc->halo.up + preLoc->halo.dn)
                                         * preLoc->nf;
         }
         int numLoop =
               activityCube.isSparse ? activityCube.numActive[b] : mPreLayer->getNumExtended();

         if (!mThreadGateIdxBuffer.empty()) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
            for (int i = 0; i < parent->getNumThreads() * getPostLayer()->getNumNeurons(); i++) {
               int ti                       = i / getPostLayer()->getNumNeurons();
               int ni                       = i % getPostLayer()->getNumNeurons();
               mThreadGateIdxBuffer[ti][ni] = -1;
            }
         }

#ifdef PV_USE_OPENMP_THREADS
         // Clear all gsyn buffers
         if (!mThreadGSyn.empty()) {
            int numNeurons = getPostLayer()->getNumNeurons();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
            for (int i = 0; i < parent->getNumThreads() * numNeurons; i++) {
               int ti              = i / numNeurons;
               int ni              = i % numNeurons;
               mThreadGSyn[ti][ni] = resetVal;
            }
         }
#endif // PV_USE_OPENMP_THREADS
         std::size_t const *gSynPatchStart = preWeights->getGeometry()->getGSynPatchStart().data();

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
         for (int loopIndex = 0; loopIndex < numLoop; loopIndex++) {
            int kPreExt;
            float a; // We never convert rates to spike counts in pooling conns
            if (activityCube.isSparse) {
               kPreExt = activeIndicesBatch[loopIndex].index;
               a       = activeIndicesBatch[loopIndex].value;
            }
            else {
               kPreExt = loopIndex;
               a       = activityBatch[kPreExt];
            }

            // If we're using mThreadGSyn, set this here
            float *gSynPatchHead;
            float *gatePatchHead = NULL;
#ifdef PV_USE_OPENMP_THREADS
            if (!mThreadGSyn.empty()) {
               int ti        = omp_get_thread_num();
               gSynPatchHead = mThreadGSyn[ti].data();
            }
            else {
               gSynPatchHead = gSynPatchHeadBatch;
            }

            if (mNeedPostIndexLayer) {
               if (!mThreadGateIdxBuffer.empty()) {
                  int ti        = omp_get_thread_num();
                  gatePatchHead = mThreadGateIdxBuffer[ti].data();
               }
               else {
                  gatePatchHead = gatePatchHeadBatch;
               }
            }
#else // PV_USE_OPENMP_THREADS
            gSynPatchHead = gSynPatchHeadBatch;
            if (mNeedPostIndexLayer) {
               gatePatchHead = gatePatchHeadBatch;
            }
#endif // PV_USE_OPENMP_THREADS
            Patch const *patch        = &preWeights->getPatch(kPreExt);
            int const nk              = patch->nx * preWeights->getPatchSizeF();
            int const ny              = patch->ny;
            int const sy              = postLoc->nx * postLoc->nf; // stride in restricted layer
            float *weightDataStart    = nullptr;
            float *postPatchStart     = &gSynPatchHead[gSynPatchStart[kPreExt]];
            float *postGatePatchStart = &gatePatchHead[gSynPatchStart[kPreExt]];

            int const kxPreExt =
                  kxPos(kPreExt,
                        preLoc->nx + preLoc->halo.lt + preLoc->halo.rt,
                        preLoc->ny + preLoc->halo.dn + preLoc->halo.up,
                        preLoc->nf);
            int const kyPreExt =
                  kyPos(kPreExt,
                        preLoc->nx + preLoc->halo.lt + preLoc->halo.rt,
                        preLoc->ny + preLoc->halo.dn + preLoc->halo.up,
                        preLoc->nf);
            int const kfPre = featureIndex(
                  kPreExt,
                  preLoc->nx + preLoc->halo.lt + preLoc->halo.rt,
                  preLoc->ny + preLoc->halo.dn + preLoc->halo.up,
                  preLoc->nf);

            int const kxPreGlobalExt = kxPreExt + preLoc->kx0;
            int const kyPreGlobalExt = kyPreExt + preLoc->ky0;

            int const kPreGlobalExt = kIndex(
                  kxPreGlobalExt,
                  kyPreGlobalExt,
                  kfPre,
                  preLoc->nxGlobal + preLoc->halo.lt + preLoc->halo.rt,
                  preLoc->nyGlobal + preLoc->halo.up + preLoc->halo.dn,
                  preLoc->nf);

            int offset   = kfPre;
            int sf       = preWeights->getPatchSizeF();
            void *auxPtr = nullptr;
            for (int y = 0; y < ny; y++) {
               if (mNeedPostIndexLayer) {
                  auxPtr = &postGatePatchStart[y * sy + offset];
               }
               (accumulateFunctionPointer)(
                     kPreGlobalExt, nk, postPatchStart + y * sy + offset, a, &w, auxPtr, sf);
            }
         }
#ifdef PV_USE_OPENMP_THREADS
         // Accumulate back into gSyn // Should this be done in HyPerLayer where it
         // can be done once,
         // as opposed to once per connection?
         if (!mThreadGSyn.empty()) {
            float *gSynPatchHead = gSynPatchHeadBatch;
            float *gateIdxBuffer = nullptr;
            if (mNeedPostIndexLayer && !mThreadGateIdxBuffer.empty()) {
               gateIdxBuffer = gatePatchHeadBatch;
            }
            int numNeurons = getPostLayer()->getNumNeurons();
// Looping over neurons first to be thread safe
#pragma omp parallel for
            for (int ni = 0; ni < numNeurons; ni++) {
               // Different for maxpooling
               if (mAccumulateType == MAXPOOLING) {
                  for (int ti = 0; ti < parent->getNumThreads(); ti++) {
                     if (gSynPatchHead[ni] < mThreadGSyn[ti][ni]) {
                        gSynPatchHead[ni] = mThreadGSyn[ti][ni];
                        if (mNeedPostIndexLayer && !mThreadGateIdxBuffer.empty()) {
                           gateIdxBuffer[ni] = mThreadGateIdxBuffer[ti][ni];
                        }
                     }
                  }
               }
               else {
                  for (int ti = 0; ti < parent->getNumThreads(); ti++) {
                     gSynPatchHead[ni] += mThreadGSyn[ti][ni];
                  }
               }
            }
         }
#endif
      }
      if (activityCube.isSparse) {
         for (int k = 0; k < getPostLayer()->getNumNeuronsAllBatches(); k++) {
            if (gSyn[k] == -INFINITY) {
               gSyn[k] = 0.0f;
            }
         }
      }
   }
}

void PoolingDelivery::clearGateIdxBuffer() {
   if (mNeedPostIndexLayer) {
      // Reset mPostIndexLayer's gsyn
      resetGSynBuffers_PoolingIndexLayer(
            mPostIndexLayer->getLayerLoc()->nbatch,
            mPostIndexLayer->getNumNeurons(),
            mPostIndexLayer->getNumChannels(),
            mPostIndexLayer->getChannel(CHANNEL_EXC));
   }
}

bool PoolingDelivery::isAllInputReady() {
   bool isReady = true;
   if (getChannelCode() != CHANNEL_NOUPDATE) {
      int const numAxonalArbors = mArborList->getNumAxonalArbors();
      for (int a = 0; a < numAxonalArbors; a++) {
         isReady &= getPreLayer()->isExchangeFinished(mArborList->getDelay(a));
      }
   }
   return isReady;
}

#ifdef PV_USE_CUDA
void PoolingDelivery::deliverGPU() {}
#endif // PV_USE_CUDA

} // end namespace PV
