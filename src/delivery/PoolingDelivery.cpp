/*
 * PoolingDelivery.cpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#include "PoolingDelivery.hpp"
#include "components/PoolingIndexLayerInputBuffer.hpp"
#include "delivery/accumulate_functions.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

PoolingDelivery::PoolingDelivery(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

PoolingDelivery::PoolingDelivery() {}

PoolingDelivery::~PoolingDelivery() {}

void PoolingDelivery::initialize(char const *name, PVParams *params, Communicator const *comm) {
   BaseDelivery::initialize(name, params, comm);
}

void PoolingDelivery::setObjectType() { mObjectType = "PoolingDelivery"; }

int PoolingDelivery::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseDelivery::ioParamsFillGroup(ioFlag);
   ioParam_pvpatchAccumulateType(ioFlag);
   ioParam_updateGSynFromPostPerspective(ioFlag);
   ioParam_needPostIndexLayer(ioFlag);
   ioParam_postIndexLayerName(ioFlag);
   return status;
}

void PoolingDelivery::ioParam_pvpatchAccumulateType(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamStringRequired(
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
   auto *params = parameters();
   pvAssert(!params->presentAndNotBeenRead(name, "receiveGpu"));
   if (!mReceiveGpu) {
      params->ioParamValue(
            ioFlag,
            name,
            "updateGSynFromPostPerspective",
            &mUpdateGSynFromPostPerspective,
            mUpdateGSynFromPostPerspective);
   }
   else {
      mUpdateGSynFromPostPerspective = true;
      params->handleUnnecessaryParameter(name, "updateGSynFromPostPerspective", true);
   }
}

void PoolingDelivery::ioParam_needPostIndexLayer(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag, name, "needPostIndexLayer", &mNeedPostIndexLayer, mNeedPostIndexLayer);
}

void PoolingDelivery::ioParam_postIndexLayerName(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "needPostIndexLayer"));
   if (mNeedPostIndexLayer) {
      parameters()->ioParamStringRequired(ioFlag, name, "postIndexLayerName", &mPostIndexLayerName);
   }
}

Response::Status
PoolingDelivery::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = BaseDelivery::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   auto *objectTable = message->mObjectTable;

   mPatchSize = objectTable->findObject<PatchSize>(getName());
   FatalIf(mPatchSize == nullptr, "%s requires a PatchSize component.\n", getDescription_c());
   if (!mPatchSize->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }

   mWeightsPair = objectTable->findObject<ImpliedWeightsPair>(getName());
   FatalIf(
         mWeightsPair == nullptr,
         "%s requires an ImpliedWeightsPair component.\n",
         getDescription_c());
   if (!mWeightsPair->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }

   if (mNeedPostIndexLayer) {
      pvAssert(mPostIndexLayerName);
      mPostIndexLayer = objectTable->findObject<PoolingIndexLayer>(mPostIndexLayerName);
   }

   if (mUpdateGSynFromPostPerspective) {
      mWeightsPair->needPost();
   }
   else {
      mWeightsPair->needPre();
   }

#ifdef PV_USE_CUDA
   if (mReceiveGpu) {
      // we need pre datastore, weights, and post gsyn for the channelCode allocated on the GPU.
      mPreData->setAllocCudaDatastore();
      mPostGSyn->useCuda();
      Weights *weights = mWeightsPair->getPostWeights();
      pvAssert(weights);
      weights->useGPU();

      // If recv from pre and pre layer is sparse, allocate activeIndices
      if (!mUpdateGSynFromPostPerspective && mPreData->getSparseLayer()) {
         mPreData->setAllocCudaActiveIndices();
      }
   }
#endif // PV_USE_CUDA
   return Response::SUCCESS;
}

#ifdef PV_USE_CUDA
Response::Status
PoolingDelivery::setCudaDevice(std::shared_ptr<SetCudaDeviceMessage const> message) {
   auto status = BaseDelivery::setCudaDevice(message);
   if (status != Response::SUCCESS) {
      return status;
   }
   if (mUsingGPUFlag) {
      Weights *weights = mWeightsPair->getPostWeights();
      pvAssert(weights);
      weights->setCudaDevice(message->mCudaDevice);
   }
   return Response::SUCCESS;
}
#endif // PV_USE_CUDA

Response::Status PoolingDelivery::allocateDataStructures() {
   if (mPostIndexLayer and !mPostIndexLayer->getDataStructuresAllocatedFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until postIndexLayer \"%s\" has finished its "
               "allocateDataStructures stage.\n",
               getDescription_c(),
               mPostIndexLayer->getName());
      }
      return Response::POSTPONE;
   }
   auto status = BaseDelivery::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   pvAssert(mPreData and mPreData->getLayerLoc());
   pvAssert(mPostGSyn and mPostGSyn->getLayerLoc());
   FatalIf(
         mPreData->getLayerLoc()->nf != mPostGSyn->getLayerLoc()->nf,
         "%s requires pre layer \"%s\" and post layer \"%s\" have equal nf (%d versus %d).\n",
         getDescription_c(),
         mPreData->getName(),
         mPostGSyn->getName(),
         mPreData->getLayerLoc()->nf,
         mPostGSyn->getLayerLoc()->nf);
#ifdef PV_USE_CUDA
   if (mReceiveGpu) {
      if (!mPreData->getDataStructuresAllocatedFlag()) {
         return Response::POSTPONE;
      }
      if (!mPostGSyn->getDataStructuresAllocatedFlag()) {
         return Response::POSTPONE;
      }
      if (!mWeightsPair->getDataStructuresAllocatedFlag()) {
         return Response::POSTPONE;
      }
      initializeDeliverKernelArgs();
   }
#endif // PV_USE_CUDA
#ifdef PV_USE_OPENMP_THREADS
   allocateThreadGSyn();
   allocateThreadGateIdxBuffer();
#endif // PV_USE_OPENMP_THREADS
   return Response::SUCCESS;
}

#ifdef PV_USE_CUDA
void PoolingDelivery::initializeDeliverKernelArgs() {
   PVCuda::CudaBuffer *d_preDatastore = mPreData->getCudaDatastore();
   PVCuda::CudaBuffer *d_postGSyn     = mPostGSyn->getCudaBuffer();
   Weights *weights                   = mWeightsPair->getPostWeights();
   pvAssert(weights);
   cudnnPoolingMode_t poolingMode = mAccumulateType == MAXPOOLING
                                          ? CUDNN_POOLING_MAX
                                          : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;

   int const nxpPost = weights->getPatchSizeX();
   int const nypPost = weights->getPatchSizeY();
   int multiplier    = mAccumulateType == SUMPOOLING ? nxpPost * nypPost : 1;

   mRecvKernel = new PVCuda::CudaPoolingDeliverKernel(mCudaDevice);
   mRecvKernel->setArgs(
         mPreData->getLayerLoc(),
         mPostGSyn->getLayerLoc(),
         nxpPost,
         nypPost,
         poolingMode,
         multiplier,
         d_preDatastore,
         d_postGSyn,
         (int)mChannelCode);
}
#endif // PV_USE_CUDA

#ifdef PV_USE_OPENMP_THREADS
void PoolingDelivery::allocateThreadGateIdxBuffer() {
   // If multithreaded, allocate a GSyn buffer for each thread, to avoid collisions.
   int const numThreads = (int)mThreadGSyn.size();
   if (numThreads > 0) {
      mThreadGateIdxBuffer.resize(numThreads);
      // mThreadGateIdxBuffer is only a buffer for one batch element. We're threading over
      // presynaptic neuron index, not batch element; so batch elements will be processed serially.
      for (int th = 0; th < numThreads; th++) {
         mThreadGateIdxBuffer[th].resize(mPostGSyn->getBufferSize());
      }
   }
}
#endif // PV_USE_OPENMP_THREADS

void PoolingDelivery::deliver(float *destBuffer) {
   // Check if we need to update based on connection's channel
   if (getChannelCode() == CHANNEL_NOUPDATE) {
      return;
   }

   if (mReceiveGpu) {
#ifdef PV_USE_CUDA
      deliverGPU(destBuffer);
#endif // PV_USE_CUDA
   }
   else {
      if (mUpdateGSynFromPostPerspective) {
         deliverPostsynapticPerspective(destBuffer);
      }
      else {
         deliverPresynapticPerspective(destBuffer);
      }
   }
}

void PoolingDelivery::deliverPostsynapticPerspective(float *destBuffer) {
   PVLayerLoc const *sourceLoc = mPreData->getLayerLoc();
   PVLayerLoc const *targetLoc = mPostGSyn->getLayerLoc();
   Weights *postWeights        = mWeightsPair->getPostWeights();

   // Slightly inefficient to define the function pointer each time deliver() is called;
   // but the real inefficiency is calling the function pointer in a tight for-loop.
   // TODO: Use templating instead of function pointer.
   void (*accumulateFunctionPointer)(
         int kPreRes, int nk, float *v, float const *a, float const *w, void *auxPtr, int sf) =
         nullptr;
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
         // UNDEFINED is the only other possible value of mAccumulateType, but the type should be
         // defined before this function is ever called.
         break;
   }

   float w = 1.0f;
   if (mAccumulateType == AVGPOOLING) {
      PVLayerLoc const *preLoc  = mPreData->getLayerLoc();
      PVLayerLoc const *postLoc = mPostGSyn->getLayerLoc();
      float relative_XScale     = (float)preLoc->nx / (float)postLoc->nx;
      float relative_YScale     = (float)preLoc->ny / (float)postLoc->ny;
      float nxp                 = (float)mPatchSize->getPatchSizeX();
      float nyp                 = (float)mPatchSize->getPatchSizeY();
      w                         = 1.0f / (nxp * relative_XScale * nyp * relative_YScale);
   }

   PVLayerCube activityCube = mPreData->getPublisher()->createCube(0 /*delay*/);

   float *gSyn = destBuffer;
   pvAssert(gSyn);

   // Get number of neurons restricted target
   int const numPostRestricted = mPostGSyn->getBufferSize();

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
      auto *indexLayerInput = mPostIndexLayer->getComponentByType<PoolingIndexLayerInputBuffer>();
      gatePatchHead         = indexLayerInput->getIndexBuffer(0);
   }

   float resetVal = 0.0f;
   if (mAccumulateType == MAXPOOLING) {
      resetVal = -INFINITY;
   }

   int const nbatch = sourceLoc->nbatch;
   FatalIf(
         targetLoc->nbatch != nbatch,
         "%s has different presynaptic and postsynaptic batch sizes.\n",
         getDescription_c());
   for (int b = 0; b < nbatch; b++) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for (int kTargetRes = 0; kTargetRes < numPostRestricted; kTargetRes++) {
         float const *activityBatch = activityCube.data
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

         float const *activityStartBuf = &(activityBatch[startSourceExt]);

         int sf           = postWeights->getPatchSizeF();
         int yPatchSize   = postWeights->getPatchSizeY();
         int numPerStride = postWeights->getPatchSizeX() * postWeights->getPatchSizeF();

         const PVLayerLoc *postLoc = mPostGSyn->getLayerLoc();
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

            float const *activityY = &(activityStartBuf[ky * sy + offset]);

            (accumulateFunctionPointer)(
                  kPreGlobalExt, numPerStride, gSynPatchPos, activityY, &w, gatePatchPos, sf);
         }
      }
   }
}

void PoolingDelivery::deliverPresynapticPerspective(float *destBuffer) {
   PVLayerLoc const *preLoc  = mPreData->getLayerLoc();
   PVLayerLoc const *postLoc = mPostGSyn->getLayerLoc();
   Weights *preWeights       = mWeightsPair->getPreWeights();

   // Slightly inefficient to define the function pointer each time deliver() is called;
   // but the real inefficiency is calling the function pointer in a tight for-loop.
   // TODO: Use templating instead of function pointer.
   void (*accumulateFunctionPointer)(
         int kPreRes, int nk, float *v, float a, float const *w, void *auxPtr, int sf) = nullptr;
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
      float relative_XScale = (float)preLoc->nx / (float)postLoc->nx;
      float relative_YScale = (float)preLoc->ny / (float)postLoc->ny;
      float nxp             = (float)mPatchSize->getPatchSizeX();
      float nyp             = (float)mPatchSize->getPatchSizeY();
      w                     = 1.0f / (nxp * relative_XScale * nyp * relative_YScale);
   }

   PVLayerCube activityCube = mPreData->getPublisher()->createCube(0 /*delay*/);

   float *gSyn = destBuffer;
   pvAssert(gSyn);

   float resetVal = 0;
   if (mAccumulateType == MAXPOOLING) {
      resetVal = -INFINITY;
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for (int i = 0; i < mPostGSyn->getBufferSizeAcrossBatch(); i++) {
         gSyn[i] = resetVal;
      }
   }

   clearGateIdxBuffer();

   for (int b = 0; b < activityCube.loc.nbatch; b++) {
      float const *activityBatch = activityCube.data
                                   + b * (preLoc->nx + preLoc->halo.rt + preLoc->halo.lt)
                                           * (preLoc->ny + preLoc->halo.up + preLoc->halo.dn)
                                           * preLoc->nf;
      float *gSynPatchHeadBatch = gSyn + b * postLoc->nx * postLoc->ny * postLoc->nf;
      float *gatePatchHeadBatch = NULL;
      if (mNeedPostIndexLayer) {
         auto *indexLayerInput =
               mPostIndexLayer->getComponentByType<PoolingIndexLayerInputBuffer>();
         gatePatchHeadBatch = indexLayerInput->getIndexBuffer(b);
      }

      SparseList<float>::Entry const *activeIndicesBatch = nullptr;
      int numLoop;
      if (activityCube.isSparse) {
         activeIndicesBatch = (SparseList<float>::Entry *)activityCube.activeIndices
                              + b * (preLoc->nx + preLoc->halo.rt + preLoc->halo.lt)
                                      * (preLoc->ny + preLoc->halo.up + preLoc->halo.dn)
                                      * preLoc->nf;
         numLoop = activityCube.numActive[b];
      }
      else {
         numLoop = activityCube.numItems / activityCube.loc.nbatch;
      }

#ifdef PV_USE_OPENMP_THREADS
      int const numThreads = (int)mThreadGateIdxBuffer.size();
      pvAssert((int)mThreadGSyn.size() == numThreads);
      if (numThreads > 0) {
#pragma omp parallel for
         for (int i = 0; i < numThreads * mPostGSyn->getBufferSize(); i++) {
            int ti                       = i / mPostGSyn->getBufferSize();
            int ni                       = i % mPostGSyn->getBufferSize();
            mThreadGateIdxBuffer[ti][ni] = -1.0f;
         }
      }
#endif // PV_USE_OPENMP_THREADS

#ifdef PV_USE_OPENMP_THREADS
      // Reset all thread gsyn buffer (note: the only difference from
      // BaseDelivery::clearThreadGSyn() is that the former sets the values to zero, while this
      // method sets them to resetVal (which is -infinity for maxpooling and zero for other types).
      if (numThreads > 0) {
         int numNeurons = mPostGSyn->getBufferSize();
         for (int ti = 0; ti < numThreads; ti++) {
            float *threadData = mThreadGSyn[ti].data();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
            for (int ni = 0; ni < numNeurons; ni++) {
               threadData[ni] = resetVal;
            }
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

         float *gSynPatchHead;
         float *gatePatchHead = NULL;
// If we're using GSyn threads, set this here
#ifdef PV_USE_OPENMP_THREADS
         if (numThreads > 1) {
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
      // Accumulate back into gSyn
      if (mAccumulateType == MAXPOOLING) {
         if (numThreads > 1) {
            int numNeurons       = mPostGSyn->getBufferSize();
            float *gateIdxBuffer = nullptr;
            if (mNeedPostIndexLayer && !mThreadGateIdxBuffer.empty()) {
               gateIdxBuffer = gatePatchHeadBatch;
            }
// Looping over neurons first to be thread safe
#pragma omp parallel for
            for (int ni = 0; ni < numNeurons; ni++) {
               for (int ti = 0; ti < numThreads; ti++) {
                  float *threadData = mThreadGSyn[ti].data();
                  if (gSynPatchHeadBatch[ni] < threadData[ni]) {
                     gSynPatchHeadBatch[ni] = threadData[ni];
                     if (gateIdxBuffer) {
                        gateIdxBuffer[ni] = mThreadGateIdxBuffer[ti][ni];
                     }
                  }
               }
            }
         }
      }
      else {
         accumulateThreadGSyn(gSynPatchHeadBatch);
      }
#endif
   }
   if (activityCube.isSparse) {
      for (int k = 0; k < mPostGSyn->getBufferSizeAcrossBatch(); k++) {
         if (gSyn[k] == -INFINITY) {
            gSyn[k] = 0.0f;
         }
      }
   }
}

void PoolingDelivery::clearGateIdxBuffer() {
   if (mNeedPostIndexLayer) {
      // Reset mPostIndexLayer's gsyn
      auto *indexLayerInput = mPostIndexLayer->getComponentByType<PoolingIndexLayerInputBuffer>();

      int const numNeuronsAcrossBatch = indexLayerInput->getBufferSizeAcrossBatch();
      int const numNeuronsAllChannels = numNeuronsAcrossBatch * indexLayerInput->getNumChannels();
      float *gSynHead                 = indexLayerInput->getIndexBuffer(0);

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
      for (int k = 0; k < numNeuronsAllChannels; k++) {
         gSynHead[k] = -1.0f;
      }
   }
}

bool PoolingDelivery::isAllInputReady() const {
   bool isReady = true;
   if (getChannelCode() != CHANNEL_NOUPDATE) {
      isReady &= mPreData->isExchangeFinished(0 /*delay*/);
   }
   return isReady;
}

#ifdef PV_USE_CUDA
void PoolingDelivery::deliverGPU(float *destBuffer) {
   pvAssert(
         getChannelCode() != CHANNEL_NOUPDATE); // Only called by deliver(), which already checked.
   pvAssert(destBuffer);

   if (mPreData->getUpdatedCudaDatastoreFlag()) {
      PVLayerCube activityCube           = mPreData->getPublisher()->createCube(0 /*delay*/);
      float const *h_preDatastore        = activityCube.data;
      PVCuda::CudaBuffer *d_preDatastore = mPreData->getCudaDatastore();
      pvAssert(d_preDatastore);
      d_preDatastore->copyToDevice(h_preDatastore);
      // Device now has updated
      mPreData->setUpdatedCudaDatastoreFlag(false);
   }

   mRecvKernel->run();
}
#endif // PV_USE_CUDA

} // end namespace PV
