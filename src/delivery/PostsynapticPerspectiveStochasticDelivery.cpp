/*
 * PostsynapticPerspectiveStochasticDelivery.cpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#include "PostsynapticPerspectiveStochasticDelivery.hpp"

namespace PV {

PostsynapticPerspectiveStochasticDelivery::PostsynapticPerspectiveStochasticDelivery(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

PostsynapticPerspectiveStochasticDelivery::PostsynapticPerspectiveStochasticDelivery() {}

PostsynapticPerspectiveStochasticDelivery::~PostsynapticPerspectiveStochasticDelivery() {
   delete mRandState;
}

void PostsynapticPerspectiveStochasticDelivery::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   mReceiveGpu = false; // If it's true, we should be using a different class.
   BaseObject::initialize(name, params, comm);
}

void PostsynapticPerspectiveStochasticDelivery::setObjectType() {
   mObjectType = "PostsynapticPerspectiveStochasticDelivery";
}

Response::Status PostsynapticPerspectiveStochasticDelivery::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = HyPerDelivery::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   // HyPerDelivery::communicateInitInfo() postpones until mWeightsPair communicates.
   pvAssert(mWeightsPair and mWeightsPair->getInitInfoCommunicatedFlag());
   if (!mWeightsPair->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   mWeightsPair->needPost();
   return Response::SUCCESS;
}

Response::Status PostsynapticPerspectiveStochasticDelivery::allocateDataStructures() {
   auto status = HyPerDelivery::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   mRandState = new Random(mPostGSyn->getLayerLoc(), false /*restricted, not extended*/);
   return Response::SUCCESS;
}

Response::Status PostsynapticPerspectiveStochasticDelivery::initializeState(
      std::shared_ptr<InitializeStateMessage const> message) {
   auto status = HyPerDelivery::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   mDeltaTimeFactor = (float)message->mDeltaTime;
   return Response::SUCCESS;
}

void PostsynapticPerspectiveStochasticDelivery::deliver(float *destBuffer) {
   // Check if we need to update based on connection's channel
   if (getChannelCode() == CHANNEL_NOUPDATE) {
      return;
   }
   pvAssert(destBuffer);

   int numAxonalArbors = mArborList->getNumAxonalArbors();
   for (int arbor = 0; arbor < numAxonalArbors; arbor++) {
      int delay                = mArborList->getDelay(arbor);
      PVLayerCube activityCube = mPreData->getPublisher()->createCube(delay);

      // Get number of neurons restricted target
      const int numPostRestricted = mPostGSyn->getBufferSize();

      const PVLayerLoc *sourceLoc = mPreData->getLayerLoc();
      const PVLayerLoc *targetLoc = mPostGSyn->getLayerLoc();

      const int sourceNx = sourceLoc->nx;
      const int sourceNy = sourceLoc->ny;
      const int sourceNf = sourceLoc->nf;
      const int targetNx = targetLoc->nx;
      const int targetNy = targetLoc->ny;
      const int targetNf = targetLoc->nf;
      const int nbatch   = targetLoc->nbatch;

      const PVHalo *sourceHalo = &sourceLoc->halo;
      const PVHalo *targetHalo = &targetLoc->halo;

      // get source layer's extended y stride
      int sy = (sourceNx + sourceHalo->lt + sourceHalo->rt) * sourceNf;

      // The start of the gsyn buffer
      float *gSynPatchHead = destBuffer;

      // Get source layer's patch y stride
      Weights *postWeights  = mWeightsPair->getPostWeights();
      int syp               = postWeights->getPatchStrideY();
      int yPatchSize        = postWeights->getPatchSizeY();
      int numPerStride      = postWeights->getPatchSizeX() * postWeights->getPatchSizeF();
      int neuronIndexStride = targetNf < 4 ? 1 : targetNf / 4;

      for (int b = 0; b < nbatch; b++) {
         int sourceNxExt       = sourceNx + sourceHalo->rt + sourceHalo->lt;
         int sourceNyExt       = sourceNy + sourceHalo->dn + sourceHalo->up;
         int sourceNumExtended = sourceNxExt * sourceNyExt * sourceNf;

         float const *activityBatch = activityCube.data + b * sourceNumExtended;
         float *gSynPatchHeadBatch  = gSynPatchHead + b * numPostRestricted;

         // Iterate over each line in the y axis, the goal is to keep weights in the cache
         for (int ky = 0; ky < yPatchSize; ky++) {
// Threading over feature was the important change that improved cache performance by
// 5-10x. dynamic scheduling also gave another performance increase over static.
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
            for (int feature = 0; feature < neuronIndexStride; feature++) {
               for (int idx = feature; idx < numPostRestricted; idx += neuronIndexStride) {
                  float *gSyn     = gSynPatchHeadBatch + idx;
                  taus_uint4 *rng = mRandState->getRNG(idx);

                  int idxExtended = kIndexExtended(
                        idx,
                        targetNx,
                        targetNy,
                        targetNf,
                        targetHalo->lt,
                        targetHalo->rt,
                        targetHalo->dn,
                        targetHalo->up);
                  int startSourceExt = postWeights->getGeometry()->getUnshrunkenStart(idxExtended);
                  float const *a     = activityBatch + startSourceExt + ky * sy;

                  int kTargetExt = kIndexExtended(
                        idx,
                        targetNx,
                        targetNy,
                        targetNf,
                        targetHalo->lt,
                        targetHalo->rt,
                        targetHalo->dn,
                        targetHalo->up);
                  float *weightBuf    = postWeights->getDataFromPatchIndex(arbor, kTargetExt);
                  float *weightValues = weightBuf + ky * syp;

                  float dv = 0.0f;
                  for (int k = 0; k < numPerStride; ++k) {
                     *rng     = cl_random_get(*rng);
                     double p = (double)rng->s0 / cl_random_max(); // 0.0 < p < 1.0
                     dv += (p < (double)(a[k] * mDeltaTimeFactor)) * weightValues[k];
                  }
                  *gSyn += dv;
               }
            }
         }
      }
   }
}

void PostsynapticPerspectiveStochasticDelivery::deliverUnitInput(float *recvBuffer) {
   // Get number of neurons restricted target
   const int numPostRestricted = mPostGSyn->getBufferSize();

   const PVLayerLoc *targetLoc = mPostGSyn->getLayerLoc();

   const int targetNx = targetLoc->nx;
   const int targetNy = targetLoc->ny;
   const int targetNf = targetLoc->nf;
   const int nbatch   = targetLoc->nbatch;

   const PVHalo *targetHalo = &targetLoc->halo;

   // Get source layer's patch y stride
   Weights *postWeights  = mWeightsPair->getPostWeights();
   int syp               = postWeights->getPatchStrideY();
   int yPatchSize        = postWeights->getPatchSizeY();
   int numPerStride      = postWeights->getPatchSizeX() * postWeights->getPatchSizeF();
   int neuronIndexStride = targetNf < 4 ? 1 : targetNf / 4;

   int numAxonalArbors = mArborList->getNumAxonalArbors();
   for (int arbor = 0; arbor < numAxonalArbors; arbor++) {
      for (int b = 0; b < nbatch; b++) {
         float *recvBatch = recvBuffer + b * numPostRestricted;

         // Iterate over each line in the y axis, the goal is to keep weights in the cache
         for (int ky = 0; ky < yPatchSize; ky++) {
// Threading over feature was the important change that improved cache performance by
// 5-10x. dynamic scheduling also gave another performance increase over static.
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
            for (int feature = 0; feature < neuronIndexStride; feature++) {
               for (int idx = feature; idx < numPostRestricted; idx += neuronIndexStride) {
                  float *recvLocation = recvBatch + idx;
                  taus_uint4 *rng     = mRandState->getRNG(idx);

                  int kTargetExt = kIndexExtended(
                        idx,
                        targetNx,
                        targetNy,
                        targetNf,
                        targetHalo->lt,
                        targetHalo->rt,
                        targetHalo->dn,
                        targetHalo->up);
                  float *weightBuf    = postWeights->getDataFromPatchIndex(arbor, kTargetExt);
                  float *weightValues = weightBuf + ky * syp;

                  float dv = 0.0f;
                  for (int k = 0; k < numPerStride; ++k) {
                     *rng     = cl_random_get(*rng);
                     double p = (double)rng->s0 / cl_random_max(); // 0.0 < p < 1.0
                     dv += (p < (double)mDeltaTimeFactor) * weightValues[k];
                  }
                  *recvLocation += mDeltaTimeFactor * dv;
               }
            }
         }
      }
   }
}

} // end namespace PV
