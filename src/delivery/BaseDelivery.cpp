/*
 * BaseDelivery.cpp
 *
 *  Created on: Nov 17, 2017
 *      Author: Pete Schultz
 */

#include "BaseDelivery.hpp"

namespace PV {

BaseDelivery::BaseDelivery(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

void BaseDelivery::initialize(char const *name, PVParams *params, Communicator const *comm) {
   LayerInputDelivery::initialize(name, params, comm);
}

void BaseDelivery::setObjectType() { mObjectType = "BaseDelivery"; }

Response::Status
BaseDelivery::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = LayerInputDelivery::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   if (mConnectionData == nullptr) {
      mConnectionData = message->mObjectTable->findObject<ConnectionData>(getName());
   }
   FatalIf(mConnectionData == nullptr, "%s could not find a ConnectionData component.\n");
   if (!mConnectionData->getInitInfoCommunicatedFlag()) {
      if (mCommunicator->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the ConnectionData component has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return Response::POSTPONE;
   }

   auto *preLayer = mConnectionData->getPre();
   pvAssert(preLayer != nullptr);
   mPreData = preLayer->getComponentByType<BasePublisherComponent>();

   auto *postLayer = mConnectionData->getPost();
   pvAssert(postLayer != nullptr);
   mPostGSyn        = postLayer->getComponentByType<LayerInputBuffer>();
   int channelAsInt = (int)getChannelCode();
   if (channelAsInt >= 0) {
      auto *postLayerInputBuffer = mPostGSyn;
      FatalIf(
            mPostGSyn == nullptr,
            "%s post layer \"%s\" does not have a LayerInputBuffer component.\n",
            getDescription_c(),
            mPostGSyn->getName());
      mPostGSyn->requireChannel(channelAsInt);
      int numChannelsCheck = postLayerInputBuffer->getNumChannels();
      FatalIf(
            numChannelsCheck <= channelAsInt,
            "%s post layer input buffer \"%s\" failed to add channel %d\n",
            getDescription_c(),
            postLayerInputBuffer->getName(),
            channelAsInt);
      postLayerInputBuffer->addDeliverySource(this);
   }

#ifdef PV_USE_CUDA
   mUsingGPUFlag = mReceiveGpu;
#endif // PV_USE_CUDA

#ifdef PV_USE_OPENMP_THREADS
   mNumThreads = message->mNumThreads;
   InfoLog() << getDescription() << " setting mNumThreads to " << mNumThreads << ".\n";
#endif // PV_USE_OPENMP_THREADS

   return Response::SUCCESS;
}

#ifdef PV_USE_OPENMP_THREADS
void BaseDelivery::allocateThreadGSyn() {
   if (getChannelCode() >= 0) {
      int numThreads = mNumThreads;
      if (numThreads > 1) {
         PVLayerLoc const *postLoc = mPostGSyn->getLayerLoc();
         // We could use mPostGSyn->getBufferSizeAcrossBatch(), but this requires
         // checking mPostGSyn->getDataStructuresAllocatedFlag().
         int const numNeuronsAllBatches = postLoc->nx * postLoc->ny * postLoc->nf * postLoc->nbatch;
         mThreadGSyn.resize(numThreads);
         for (auto &th : mThreadGSyn) {
            th.resize(numNeuronsAllBatches);
         }
      }
   }
}

void BaseDelivery::clearThreadGSyn() {
   if (getChannelCode() >= 0) {
      int const numThreads = (int)mThreadGSyn.size();
      if (numThreads > 1) {
         int const numPostRestricted = mPostGSyn->getBufferSize();
#pragma omp parallel for schedule(static)
         for (int ti = 0; ti < numThreads; ++ti) {
            float *threadData = mThreadGSyn[ti].data();
            for (int ni = 0; ni < numPostRestricted; ++ni) {
               threadData[ni] = 0.0f;
            }
         }
      }
   }
   // Would it be better to have the pragma omp parallel on the inner loop? PoolingDelivery is
   // organized that way; and TransposePoolingDelivery used to, before it called this method.
}
#endif // PV_USE_OPENMP_THREADS

void BaseDelivery::accumulateThreadGSyn(float *baseGSynBuffer) {
#ifdef PV_USE_OPENMP_THREADS
   if (getChannelCode() >= 0) {
      int const numThreads = (int)mThreadGSyn.size();
      if (numThreads > 1) {
         int numNeuronsPost = mPostGSyn->getBufferSize();
         for (int ti = 0; ti < numThreads; ti++) {
            float *threadData = mThreadGSyn[ti].data();
// Looping over neurons is thread safe
#pragma omp parallel for
            for (int ni = 0; ni < numNeuronsPost; ni++) {
               baseGSynBuffer[ni] += threadData[ni];
            }
         }
      }
   }
#endif // PV_USE_OPENMP_THREADS
}

float *BaseDelivery::setWorkingGSynBuffer(float *baseGSynBuffer) {
   float *workingGSynBuffer = baseGSynBuffer;
#ifdef PV_USE_OPENMP_THREADS
   if (!mThreadGSyn.empty()) {
      workingGSynBuffer = mThreadGSyn[omp_get_thread_num()].data();
   }
#endif // PV_USE_OPENMP_THREADS
   return workingGSynBuffer;
}

} // namespace PV
