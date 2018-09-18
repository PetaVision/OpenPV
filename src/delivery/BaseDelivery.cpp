/*
 * BaseDelivery.cpp
 *
 *  Created on: Nov 17, 2017
 *      Author: Pete Schultz
 */

#include "BaseDelivery.hpp"
#include "columns/HyPerCol.hpp"
#include "layers/HyPerLayer.hpp"
#include "utils/MapLookupByType.hpp"

namespace PV {

BaseDelivery::BaseDelivery(char const *name, HyPerCol *hc) { initialize(name, hc); }

int BaseDelivery::initialize(char const *name, HyPerCol *hc) {
   return LayerInputDelivery::initialize(name, hc);
}

void BaseDelivery::setObjectType() { mObjectType = "BaseDelivery"; }

Response::Status
BaseDelivery::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = LayerInputDelivery::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   if (mConnectionData == nullptr) {
      mConnectionData = mapLookupByType<ConnectionData>(message->mHierarchy);
   }
   FatalIf(mConnectionData == nullptr, "%s could not find a ConnectionData component.\n");
   if (!mConnectionData->getInitInfoCommunicatedFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until the ConnectionData component has finished its "
               "communicateInitInfo stage.\n",
               getDescription_c());
      }
      return Response::POSTPONE;
   }

   mPreLayer  = mConnectionData->getPre();
   mPostLayer = mConnectionData->getPost();
   pvAssert(mPreLayer != nullptr and mPostLayer != nullptr);

   int numChannelsCheck = 0;
   int channelAsInt     = (int)getChannelCode();
   if (channelAsInt >= 0) {
      int status = getPostLayer()->requireChannel(channelAsInt, &numChannelsCheck);
      if (status != PV_SUCCESS) {
         if (parent->getCommunicator()->globalCommRank() == 0) {
            ErrorLog().printf(
                  "%s: postsynaptic layer \"%s\" failed to add channel %d\n",
                  getDescription_c(),
                  getPostLayer()->getName(),
                  channelAsInt);
         }
         MPI_Barrier(parent->getCommunicator()->globalCommunicator());
         exit(EXIT_FAILURE);
      }
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
   int const numThreads = mNumThreads;
   if (numThreads > 1) {
      int const numNeuronsAllBatches = mPostLayer->getNumNeuronsAllBatches();
      mThreadGSyn.resize(numThreads);
      for (auto &th : mThreadGSyn) {
         th.resize(numNeuronsAllBatches);
      }
   }
}

void BaseDelivery::clearThreadGSyn() {
   int const numThreads = (int)mThreadGSyn.size();
   if (numThreads > 1) {
      int const numPostRestricted = mPostLayer->getNumNeurons();
#pragma omp parallel for schedule(static)
      for (int ti = 0; ti < numThreads; ++ti) {
         float *threadData = mThreadGSyn[ti].data();
         for (int ni = 0; ni < numPostRestricted; ++ni) {
            threadData[ni] = 0.0f;
         }
      }
   }
   // Would it be better to have the pragma omp parallel on the inner loop? PoolingDelivery is
   // organized that way; and TransposePoolingDelivery used to, before it called this method.
}
#endif // PV_USE_OPENMP_THREADS

void BaseDelivery::accumulateThreadGSyn(float *baseGSynBuffer) {
#ifdef PV_USE_OPENMP_THREADS
   int const numThreads = (int)mThreadGSyn.size();
   if (numThreads > 0) {
      float *postChannel = mPostLayer->getChannel(getChannelCode());
      int numNeuronsPost = mPostLayer->getNumNeurons();
      for (int ti = 0; ti < numThreads; ti++) {
         float *threadData = mThreadGSyn[ti].data();
// Looping over neurons is thread safe
#pragma omp parallel for
         for (int ni = 0; ni < numNeuronsPost; ni++) {
            baseGSynBuffer[ni] += threadData[ni];
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
