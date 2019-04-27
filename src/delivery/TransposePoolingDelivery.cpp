/*
 * TransposePoolingDelivery.cpp
 *
 *  Created on: Jan 9, 2018
 *      Author: Pete Schultz
 */

#include "TransposePoolingDelivery.hpp"
#include "components/OriginalConnNameParam.hpp"
#include "connections/PoolingConn.hpp"
#include "delivery/accumulate_functions.hpp"
#include "observerpattern/ObserverTable.hpp"

namespace PV {

TransposePoolingDelivery::TransposePoolingDelivery(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

TransposePoolingDelivery::TransposePoolingDelivery() {}

TransposePoolingDelivery::~TransposePoolingDelivery() {}

void TransposePoolingDelivery::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   BaseDelivery::initialize(name, params, comm);
}

void TransposePoolingDelivery::setObjectType() { mObjectType = "TransposePoolingDelivery"; }

int TransposePoolingDelivery::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseDelivery::ioParamsFillGroup(ioFlag);
   ioParam_updateGSynFromPostPerspective(ioFlag);
   return status;
}

void TransposePoolingDelivery::ioParam_receiveGpu(enum ParamsIOFlag ioFlag) {
   // During the communication phase, receiveGpu will be copied from the original conn
   if (ioFlag == PARAMS_IO_READ) {
      parameters()->handleUnnecessaryParameter(name, "receiveGpu");
   }
}

void TransposePoolingDelivery::ioParam_updateGSynFromPostPerspective(enum ParamsIOFlag ioFlag) {
   // To read this param, we need to wait until the CommunicateInitInfo stage, because the behavior
   // depends on mReceiveGpu, which isn't determined until the communicate stage, since it is
   // copied from the original conn.
   if (ioFlag == PARAMS_IO_WRITE) {
      if (!mReceiveGpu) {
         parameters()->ioParamValue(
               ioFlag,
               name,
               "updateGSynFromPostPerspective",
               &mUpdateGSynFromPostPerspective,
               mUpdateGSynFromPostPerspective);
      }
   }
}

Response::Status TransposePoolingDelivery::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = BaseDelivery::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   auto *objectTable = message->mObjectTable;

   auto *originalConnNameParam = objectTable->findObject<OriginalConnNameParam>(getName());
   pvAssert(originalConnNameParam);
   if (!originalConnNameParam->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   const char *originalConnName = originalConnNameParam->getLinkedObjectName();

   auto *originalPoolingDelivery = objectTable->findObject<PoolingDelivery>(originalConnName);
   FatalIf(
         originalPoolingDelivery == nullptr,
         "%s: original connection \"%s\" does not have a PoolingDelivery component.\n",
         getDescription_c(),
         originalConnName);
   mAccumulateType = originalPoolingDelivery->getAccumulateType();
   mReceiveGpu     = originalPoolingDelivery->getReceiveGpu();
#ifdef PV_USE_CUDA
   mUsingGPUFlag = originalPoolingDelivery->isUsingGPU();
#endif // PV_USE_CUDA
   auto *originalPostIndexLayer = originalPoolingDelivery->getPostIndexLayer();
   if (originalPostIndexLayer) {
      mOriginalPostIndexData = originalPostIndexLayer->getComponentByType<BasePublisherComponent>();
   }

   auto *originalConnectionData = objectTable->findObject<ConnectionData>(originalConnName);
   FatalIf(
         originalConnectionData == nullptr,
         "%s: original connection \"%s\" does not have a ConnectionData component.\n",
         getDescription_c(),
         originalConnName);
   mOriginalPreData =
         originalConnectionData->getPre()->getComponentByType<BasePublisherComponent>();
   mOriginalPostGSyn = originalConnectionData->getPost()->getComponentByType<LayerInputBuffer>();

   // If receiveGpu is false, we need to read updateGSynFromPostPerspective.
   // If it is true, we use the CUDA routine, which always uses the post perspective.
   if (!mReceiveGpu) {
      parameters()->ioParamValue(
            PARAMS_IO_READ,
            name,
            "updateGSynFromPostPerspective",
            &mUpdateGSynFromPostPerspective,
            mUpdateGSynFromPostPerspective);
   }
   else {
      mUpdateGSynFromPostPerspective = true;
      parameters()->handleUnnecessaryParameter(
            name, "updateGSynFromPostPerspective", mUpdateGSynFromPostPerspective);
   }

   mPatchSize = objectTable->findObject<DependentPatchSize>(getName());
   FatalIf(
         mPatchSize == nullptr,
         "%s requires a DependentPatchSize component.\n",
         getDescription_c());
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

   if (mUpdateGSynFromPostPerspective) {
      mWeightsPair->needPost();
   }
   else {
      mWeightsPair->needPre();
   }

#ifdef PV_USE_CUDA
   if (mReceiveGpu) {
      // we need pre datastore, weights, and post gsyn for the channelCode allocated on the GPU.
      getPreData()->setAllocCudaDatastore();
      mPostGSyn->useCuda();
      Weights *weights = mWeightsPair->getPostWeights();
      pvAssert(weights);
      weights->useGPU();

      // If recv from pre and pre layer is sparse, allocate activeIndices
      if (!mUpdateGSynFromPostPerspective && getPreData()->getSparseLayer()) {
         getPreData()->setAllocCudaActiveIndices();
      }
   }
#endif // PV_USE_CUDA

   return Response::SUCCESS;
}

#ifdef PV_USE_CUDA
Response::Status
TransposePoolingDelivery::setCudaDevice(std::shared_ptr<SetCudaDeviceMessage const> message) {
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

Response::Status TransposePoolingDelivery::allocateDataStructures() {
   auto status = BaseDelivery::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   pvAssert(mPreData and mPreData->getLayerLoc());
   pvAssert(mPostGSyn and mPostGSyn->getLayerLoc());
   FatalIf(
         mPreData->getLayerLoc()->nf != mPostGSyn->getLayerLoc()->nf,
         "%s pre layer \"%s\" has %d features but post layer \"%s\" has %d features. "
         "The numbers of features must be equal.\n",
         getDescription_c(),
         mPreData->getName(),
         mPreData->getLayerLoc()->nf,
         mPostGSyn->getName(),
         mPostGSyn->getLayerLoc()->nf);
#ifdef PV_USE_CUDA
   if (mReceiveGpu) {
      if (!mPreData->getDataStructuresAllocatedFlag()) {
         return Response::POSTPONE;
      }
      if (!mPostGSyn->getDataStructuresAllocatedFlag()) {
         return Response::POSTPONE;
      }
      if (!mOriginalPreData->getDataStructuresAllocatedFlag()) {
         return Response::POSTPONE;
      }
      if (!mOriginalPostGSyn->getDataStructuresAllocatedFlag()) {
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
#endif // PV_USE_OPENMP_THREADS
   return Response::SUCCESS;
}

#ifdef PV_USE_CUDA
void TransposePoolingDelivery::initializeDeliverKernelArgs() {
   PVCuda::CudaBuffer *d_preDatastore         = mPreData->getCudaDatastore();
   PVCuda::CudaBuffer *d_postGSyn             = mPostGSyn->getCudaBuffer();
   PVCuda::CudaBuffer *d_originalPreDatastore = mOriginalPreData->getCudaDatastore();
   PVCuda::CudaBuffer *d_originalPostGSyn     = mOriginalPostGSyn->getCudaBuffer();
   Weights *weights                           = mWeightsPair->getPostWeights();
   pvAssert(weights);
   int const nxpPost = weights->getPatchSizeX();
   int const nypPost = weights->getPatchSizeY();
   int multiplier    = mAccumulateType == PoolingDelivery::SUMPOOLING ? nxpPost * nypPost : 1;

   cudnnPoolingMode_t poolingMode = mAccumulateType == PoolingDelivery::MAXPOOLING
                                          ? CUDNN_POOLING_MAX
                                          : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
   mDeliverKernel = new PVCuda::CudaTransposePoolingDeliverKernel(mCudaDevice);
   mDeliverKernel->setArgs(
         mPreData->getLayerLoc(),
         mPostGSyn->getLayerLoc(),
         mOriginalPreData->getLayerLoc(),
         mOriginalPostGSyn->getLayerLoc(),
         nxpPost,
         nypPost,
         poolingMode,
         multiplier,
         d_preDatastore,
         d_postGSyn,
         d_originalPreDatastore,
         d_originalPostGSyn,
         (int)mChannelCode);
}
#endif // PV_USE_CUDA

void TransposePoolingDelivery::deliver(float *destBuffer) {
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

void TransposePoolingDelivery::deliverPostsynapticPerspective(float *destBuffer) {
   Fatal() << "Delivering from PostSynapticPerspective for TransposePoolingDelivery has not been "
              "implemented yet.\n";
}

void TransposePoolingDelivery::deliverPresynapticPerspective(float *destBuffer) {
   PVLayerLoc const *preLoc  = mPreData->getLayerLoc();
   PVLayerLoc const *postLoc = mPostGSyn->getLayerLoc();
   Weights *preWeights       = mWeightsPair->getPreWeights();

   PVLayerCube activityCube = mPreData->getPublisher()->createCube(0 /*delay*/);

   float *gSyn = destBuffer;
   pvAssert(gSyn);

   // Grab postIdxLayer's data
   float const *postIdxData = nullptr;
   int postIdxNumExtended   = 0;
   if (mAccumulateType == PoolingDelivery::MAXPOOLING) {
      pvAssert(mOriginalPostIndexData);
      PVLayerCube cube   = mOriginalPostIndexData->getPublisher()->createCube(0 /*delay*/);
      postIdxData        = cube.data;
      postIdxNumExtended = mOriginalPostIndexData->getNumExtended();
   }

   int const nbatch = preLoc->nbatch;
   FatalIf(
         postLoc->nbatch != nbatch,
         "%s has different presynaptic and postsynaptic batch sizes.\n",
         getDescription_c());
   for (int b = 0; b < nbatch; b++) {
      float const *activityBatch    = activityCube.data + b * mPreData->getNumExtended();
      float *gSynPatchHeadBatch     = gSyn + b * postLoc->nx * postLoc->ny * postLoc->nf;
      float const *postIdxDataBatch = nullptr;
      if (mAccumulateType == PoolingDelivery::MAXPOOLING) {
         postIdxDataBatch = postIdxData + b * postIdxNumExtended;
      }

      SparseList<float>::Entry const *activeIndicesBatch = nullptr;
      int numLoop;
      if (activityCube.isSparse) {
         activeIndicesBatch = (SparseList<float>::Entry *)activityCube.activeIndices
                              + b * mPreData->getNumExtended();
         numLoop = activityCube.numActive[b];
      }
      else {
         numLoop = activityCube.numItems / activityCube.loc.nbatch;
      }

#ifdef PV_USE_OPENMP_THREADS
      // Note: before the clearThreadGSyn method was abstracted out, PoolingDelivery did an
      // equivalent task here, but parallelized on the inner for-loop instead of parallelizing on
      // the outer loop with the schedule(static) directive. Does this affect performance?
      clearThreadGSyn();
#endif // PV_USE_OPENMP_THREADS
      std::size_t const *gSynPatchStart = preWeights->getGeometry()->getGSynPatchStart().data();

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
      for (int loopIndex = 0; loopIndex < numLoop; loopIndex++) {
         float a     = 0.0f;
         int kPreExt = loopIndex;
         if (activityCube.isSparse) {
            a       = activeIndicesBatch[loopIndex].value;
            kPreExt = activeIndicesBatch[loopIndex].index;
         }
         else {
            a = activityBatch[loopIndex];
         }
         if (a == 0.0f) {
            continue;
         }

         float *gSynPatchHead;
// If we're using threaded GSyn, set this here
#ifdef PV_USE_OPENMP_THREADS
         if (!mThreadGSyn.empty()) {
            int ti        = omp_get_thread_num();
            gSynPatchHead = mThreadGSyn[ti].data();
         }
         else {
            gSynPatchHead = gSynPatchHeadBatch;
         }
#else // PV_USE_OPENMP_THREADS
         gSynPatchHead = gSynPatchHeadBatch;
#endif // PV_USE_OPENMP_THREADS

         const int kxPreExt =
               kxPos(kPreExt,
                     preLoc->nx + preLoc->halo.lt + preLoc->halo.rt,
                     preLoc->ny + preLoc->halo.dn + preLoc->halo.up,
                     preLoc->nf);
         const int kyPreExt =
               kyPos(kPreExt,
                     preLoc->nx + preLoc->halo.lt + preLoc->halo.rt,
                     preLoc->ny + preLoc->halo.dn + preLoc->halo.up,
                     preLoc->nf);
         const int kfPre = featureIndex(
               kPreExt,
               preLoc->nx + preLoc->halo.lt + preLoc->halo.rt,
               preLoc->ny + preLoc->halo.dn + preLoc->halo.up,
               preLoc->nf);

         if (mAccumulateType == PoolingDelivery::MAXPOOLING) {
            const int kxPreGlobalExt = kxPreExt + preLoc->kx0;
            const int kyPreGlobalExt = kyPreExt + preLoc->ky0;
            if (kxPreGlobalExt < preLoc->halo.lt
                || kxPreGlobalExt >= preLoc->nxGlobal + preLoc->halo.lt
                || kyPreGlobalExt < preLoc->halo.up
                || kyPreGlobalExt >= preLoc->nyGlobal + preLoc->halo.up) {
               continue;
            }

            // Convert stored global extended index into local extended index
            int postGlobalExtIdx = (int)postIdxDataBatch[kPreExt];

            // If all inputs are zero and input layer is sparse, postGlobalExtIdx will still be
            // -1.
            if (postGlobalExtIdx == -1) {
               continue;
            }

            // Make sure the index is in bounds
            assert(
                  postGlobalExtIdx >= 0
                  && postGlobalExtIdx
                           < (postLoc->nxGlobal + postLoc->halo.lt + postLoc->halo.rt)
                                   * (postLoc->nyGlobal + postLoc->halo.up + postLoc->halo.dn)
                                   * postLoc->nf);

            const int kxPostGlobalExt =
                  kxPos(postGlobalExtIdx,
                        postLoc->nxGlobal + postLoc->halo.lt + postLoc->halo.rt,
                        postLoc->nyGlobal + postLoc->halo.dn + postLoc->halo.up,
                        postLoc->nf);
            const int kyPostGlobalExt =
                  kyPos(postGlobalExtIdx,
                        postLoc->nxGlobal + postLoc->halo.lt + postLoc->halo.rt,
                        postLoc->nyGlobal + postLoc->halo.dn + postLoc->halo.up,
                        postLoc->nf);
            const int kfPost = featureIndex(
                  postGlobalExtIdx,
                  postLoc->nxGlobal + postLoc->halo.lt + postLoc->halo.rt,
                  postLoc->nyGlobal + postLoc->halo.dn + postLoc->halo.up,
                  postLoc->nf);

            const int kxPostLocalRes = kxPostGlobalExt - postLoc->kx0 - postLoc->halo.lt;
            const int kyPostLocalRes = kyPostGlobalExt - postLoc->ky0 - postLoc->halo.up;
            if (kxPostLocalRes < 0 || kxPostLocalRes >= postLoc->nx || kyPostLocalRes < 0
                || kyPostLocalRes >= postLoc->ny) {
               continue;
            }

            const int kPostLocalRes = kIndex(
                  kxPostLocalRes, kyPostLocalRes, kfPost, postLoc->nx, postLoc->ny, postLoc->nf);
            if (fabs(a) > fabs(gSynPatchHead[kPostLocalRes])) {
               gSynPatchHead[kPostLocalRes] = a;
            }
         }
         else {
            Patch const *patch    = &preWeights->getPatch(kPreExt);
            const int nk          = patch->nx * preWeights->getPatchSizeF();
            const int ny          = patch->ny;
            const int sy          = postLoc->nx * postLoc->nf; // stride in restricted layer
            float *postPatchStart = &gSynPatchHead[gSynPatchStart[kPreExt]];

            int offset = kfPre;
            int sf     = preWeights->getPatchSizeF();

            float w      = 1.0f;
            void *auxPtr = NULL;
            for (int y = 0; y < ny; y++) {
               pvpatch_max_pooling(0, nk, postPatchStart + y * sy + offset, a, &w, auxPtr, sf);
            }
         }
      }

#ifdef PV_USE_OPENMP_THREADS
      // Set back into gSyn
      int const numThreads = (int)mThreadGSyn.size();
      if (numThreads > 1) {
         float *gSynPatchHead = gSynPatchHeadBatch;
         int numNeurons       = mPostGSyn->getBufferSize();
// Looping over neurons first to be thread safe
#pragma omp parallel for
         for (int ni = 0; ni < numNeurons; ni++) {
            if (mAccumulateType == PoolingDelivery::MAXPOOLING) {
               // Grab maxumum magnitude of this thread's ThreadGSyn and set that value
               float maxMag  = -INFINITY;
               int maxMagIdx = -1;
               for (int ti = 0; ti < numThreads; ti++) {
                  float *threadData = mThreadGSyn[ti].data();
                  if (maxMag < fabsf(threadData[ni])) {
                     maxMag    = fabsf(threadData[ni]);
                     maxMagIdx = ti;
                  }
               }
               pvAssert(maxMagIdx >= 0);
               gSynPatchHead[ni] = mThreadGSyn[maxMagIdx].data()[ni];
            }
            else {
               for (int ti = 0; ti < numThreads; ti++) {
                  gSynPatchHead[ni] += mThreadGSyn[ti].data()[ni];
               }
            }
         }
      }
#endif
   }
}

bool TransposePoolingDelivery::isAllInputReady() const {
   bool isReady = true;
   if (getChannelCode() != CHANNEL_NOUPDATE) {
      isReady &= getPreData()->isExchangeFinished(0 /*delay*/);
   }
   return isReady;
}

#ifdef PV_USE_CUDA
void TransposePoolingDelivery::deliverGPU(float *destBuffer) {
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

   mDeliverKernel->run();
}
#endif // PV_USE_CUDA

} // end namespace PV
