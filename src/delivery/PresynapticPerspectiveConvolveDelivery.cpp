/*
 * PresynapticPerspectiveConvolveDelivery.cpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#include "PresynapticPerspectiveConvolveDelivery.hpp"
#include "structures/Weights.hpp"

namespace PV {

PresynapticPerspectiveConvolveDelivery::PresynapticPerspectiveConvolveDelivery(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

PresynapticPerspectiveConvolveDelivery::PresynapticPerspectiveConvolveDelivery() {}

PresynapticPerspectiveConvolveDelivery::~PresynapticPerspectiveConvolveDelivery() {}

void PresynapticPerspectiveConvolveDelivery::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   mReceiveGpu = false; // If it's true, we should be using a different class.
   HyPerDelivery::initialize(name, params, comm);
}

void PresynapticPerspectiveConvolveDelivery::setObjectType() {
   mObjectType = "PresynapticPerspectiveConvolveDelivery";
}

Response::Status PresynapticPerspectiveConvolveDelivery::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = HyPerDelivery::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   if (getChannelCode() == CHANNEL_NOUPDATE) { return status; }
   // HyPerDelivery::communicateInitInfo() postpones until mWeightsPair communicates.
   pvAssert(mWeightsPair and mWeightsPair->getInitInfoCommunicatedFlag());
   mWeightsPair->needPre();

   return Response::SUCCESS;
}

Response::Status PresynapticPerspectiveConvolveDelivery::allocateDataStructures() {
   auto status = HyPerDelivery::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   if (getChannelCode() == CHANNEL_NOUPDATE) { return status; }
#ifdef PV_USE_OPENMP_THREADS
   allocateThreadGSyn();
#endif // PV_USE_OPENMP_THREADS
   return Response::SUCCESS;
}

void PresynapticPerspectiveConvolveDelivery::deliver(float *destBuffer) {
   // Check if we need to update based on connection's channel
   if (getChannelCode() == CHANNEL_NOUPDATE) {
      return;
   }
   float *postChannel = destBuffer;
   pvAssert(postChannel);

   PVLayerLoc const *preLoc  = mPreData->getLayerLoc();
   PVLayerLoc const *postLoc = mPostGSyn->getLayerLoc();
   Weights *weights          = mWeightsPair->getPreWeights();

   int const nxPreExtended   = preLoc->nx + preLoc->halo.rt + preLoc->halo.rt;
   int const nyPreExtended   = preLoc->ny + preLoc->halo.dn + preLoc->halo.up;
   long const numPreExtended = (long)nxPreExtended * (long)nyPreExtended * (long)preLoc->nf;

   long const numPostRestricted = (long)postLoc->nx * (long)postLoc->ny * (long)postLoc->nf;

   int nbatch = preLoc->nbatch;
   pvAssert(nbatch == postLoc->nbatch);

   long const sy  = (long)postLoc->nx * (long)postLoc->nf; // stride in restricted layer
   long const syw = (long)weights->getGeometry()->getPatchStrideY(); // stride in patch

   bool const preLayerIsSparse = mPreData->getSparseLayerFlag();

   int numAxonalArbors = mArborList->getNumAxonalArbors();
   for (int arbor = 0; arbor < numAxonalArbors; arbor++) {
      int delay                = mArborList->getDelay(arbor);
      PVLayerCube activityCube = mPreData->getPublisher()->createCube(delay);

      for (int b = 0; b < nbatch; b++) {
         size_t batchOffset                                 = b * numPreExtended;
         float const *activityBatch                         = &activityCube.data[batchOffset];
         float *gSynPatchHeadBatch                          = postChannel + b * numPostRestricted;
         SparseList<float>::Entry const *activeIndicesBatch = nullptr;
         long numNeurons;
         if (preLayerIsSparse) {
            activeIndicesBatch =
                  (SparseList<float>::Entry *)activityCube.activeIndices + batchOffset;
            numNeurons = activityCube.numActive[b];
         }
         else {
            numNeurons = activityCube.numItems / activityCube.loc.nbatch;
         }

#ifdef PV_USE_OPENMP_THREADS
         clearThreadGSyn();
#endif

         std::size_t const *gSynPatchStart = weights->getGeometry()->getGSynPatchStart().data();
         if (!preLayerIsSparse) {
            for (int y = 0; y < weights->getPatchSizeY(); y++) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(guided)
#endif
               for (long idx = 0; idx < numNeurons; idx++) {
                  long kPreExt = idx;

                  // Weight
                  Patch const *patch = &weights->getPatch(kPreExt);

                  if (y >= patch->ny) {
                     continue;
                  }

                  // Activity
                  float a = activityBatch[kPreExt];
                  if (a == 0.0f) {
                     continue;
                  }
                  a *= mDeltaTimeFactor;

                  float *gSynPatchHead = setWorkingGSynBuffer(gSynPatchHeadBatch);

                  float *postPatchStart = &gSynPatchHead[gSynPatchStart[kPreExt]];

                  const int nk                 = patch->nx * weights->getPatchSizeF();
                  float const *weightDataHead  = weights->getDataFromPatchIndex(arbor, kPreExt);
                  float const *weightDataStart = &weightDataHead[patch->offset];

                  float *v                  = &postPatchStart[y * sy];
                  float const *weightValues = &weightDataStart[y * syw];
                  for (int k = 0; k < nk; k++) {
                     v[k] += a * weightValues[k];
                  }
               }
            }
         }
         else { // Sparse, use the stored activity / index pairs
            int const nyp = weights->getPatchSizeY();
            for (int y = 0; y < nyp; y++) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(guided)
#endif
               for (long idx = 0; idx < numNeurons; idx++) {
                  long kPreExt = activeIndicesBatch[idx].index;

                  // Weight
                  Patch const *patch = &weights->getPatch(kPreExt);

                  if (y >= patch->ny) {
                     continue;
                  }

                  // Activity
                  float a = activeIndicesBatch[idx].value;
                  if (a == 0.0f) {
                     continue;
                  }
                  a *= mDeltaTimeFactor;

                  float *gSynPatchHead = setWorkingGSynBuffer(gSynPatchHeadBatch);

                  float *postPatchStart = &gSynPatchHead[gSynPatchStart[kPreExt]];

                  const int nk                 = patch->nx * weights->getPatchSizeF();
                  float const *weightDataHead  = weights->getDataFromPatchIndex(arbor, kPreExt);
                  float const *weightDataStart = &weightDataHead[patch->offset];

                  float *v                  = &postPatchStart[y * sy];
                  float const *weightValues = &weightDataStart[y * syw];
                  for (int k = 0; k < nk; k++) {
                     v[k] += a * weightValues[k];
                  }
               }
            }
         }
         accumulateThreadGSyn(gSynPatchHeadBatch);
      } // Loop over batch elements
   } // Loop over arbors
}

void PresynapticPerspectiveConvolveDelivery::deliverUnitInput(float *recvBuffer) {
   // Check if we need to update based on connection's channel
   if (getChannelCode() == CHANNEL_NOUPDATE) {
      return;
   }
   PVLayerLoc const *preLoc = mPreData->getLayerLoc();
   int const nxPreExt       = preLoc->nx + preLoc->halo.lt + preLoc->halo.rt;
   int const nyPreExt       = preLoc->ny + preLoc->halo.dn + preLoc->halo.up;
   long const numPreExt     = (long)nxPreExt * (long)nyPreExt * (long)preLoc->nf;

   PVLayerLoc const *postLoc    = mPostGSyn->getLayerLoc();
   long const numPostRestricted = (long)postLoc->nx * (long)postLoc->ny * (long)postLoc->nf;
   int nbatch                   = postLoc->nbatch;
   long const sy                = (long)postLoc->nx * (long)postLoc->nf; // stride in restricted layer

   Weights *weights = mWeightsPair->getPreWeights();
   long const syw   = (long)weights->getGeometry()->getPatchStrideY(); // stride in patch

   int numAxonalArbors = mArborList->getNumAxonalArbors();
   for (int arbor = 0; arbor < numAxonalArbors; arbor++) {
      for (int b = 0; b < nbatch; b++) {
         float *recvBatch = &recvBuffer[b * numPostRestricted];

#ifdef PV_USE_OPENMP_THREADS
         clearThreadGSyn();
#endif

         std::size_t const *gSynPatchStart = weights->getGeometry()->getGSynPatchStart().data();
         for (int y = 0; y < weights->getPatchSizeY(); y++) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(guided)
#endif
            for (long idx = 0; idx < numPreExt; idx++) {
               long kPreExt = idx;

               // Weight
               Patch const *patch = &weights->getPatch(kPreExt);

               if (y >= patch->ny) {
                  continue;
               }

               float *recvPatchHead = setWorkingGSynBuffer(recvBatch);

               float *postPatchStart = &recvPatchHead[gSynPatchStart[kPreExt]];

               const int nk                 = patch->nx * weights->getPatchSizeF();
               float const *weightDataHead  = weights->getDataFromPatchIndex(arbor, kPreExt);
               float const *weightDataStart = &weightDataHead[patch->offset];

               float *v                  = &postPatchStart[y * sy];
               float const *weightValues = &weightDataStart[y * syw];
               for (int k = 0; k < nk; k++) {
                  v[k] += mDeltaTimeFactor * weightValues[k];
               }
            }
         }
         accumulateThreadGSyn(recvBatch);
      } // Loop over batch elements
   } // Loop over arbors
}

} // end namespace PV
