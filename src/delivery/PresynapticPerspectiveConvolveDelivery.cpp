/*
 * PresynapticPerspectiveConvolveDelivery.cpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#include "PresynapticPerspectiveConvolveDelivery.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

PresynapticPerspectiveConvolveDelivery::PresynapticPerspectiveConvolveDelivery(
      char const *name,
      HyPerCol *hc) {
   initialize(name, hc);
}

PresynapticPerspectiveConvolveDelivery::PresynapticPerspectiveConvolveDelivery() {}

PresynapticPerspectiveConvolveDelivery::~PresynapticPerspectiveConvolveDelivery() {}

int PresynapticPerspectiveConvolveDelivery::initialize(char const *name, HyPerCol *hc) {
   return BaseObject::initialize(name, hc);
}

void PresynapticPerspectiveConvolveDelivery::setObjectType() {
   mObjectType = "PresynapticPerspectiveConvolveDelivery";
}

int PresynapticPerspectiveConvolveDelivery::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerDelivery::ioParamsFillGroup(ioFlag);
   return status;
}

void PresynapticPerspectiveConvolveDelivery::ioParam_receiveGpu(enum ParamsIOFlag ioFlag) {
   mReceiveGpu = false; // If it's true, we should be using a different class.
}

Response::Status PresynapticPerspectiveConvolveDelivery::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = HyPerDelivery::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
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
#ifdef PV_USE_OPENMP_THREADS
   allocateThreadGSyn();
#endif // PV_USE_OPENMP_THREADS
   return Response::SUCCESS;
}

void PresynapticPerspectiveConvolveDelivery::deliver() {
   // Check if we need to update based on connection's channel
   if (getChannelCode() == CHANNEL_NOUPDATE) {
      return;
   }
   float *postChannel = mPostLayer->getChannel(getChannelCode());
   pvAssert(postChannel);

   PVLayerLoc const *preLoc  = mPreLayer->getLayerLoc();
   PVLayerLoc const *postLoc = mPostLayer->getLayerLoc();
   Weights *weights          = mWeightsPair->getPreWeights();

   int const nxPreExtended  = preLoc->nx + preLoc->halo.rt + preLoc->halo.rt;
   int const nyPreExtended  = preLoc->ny + preLoc->halo.dn + preLoc->halo.up;
   int const numPreExtended = nxPreExtended * nyPreExtended * preLoc->nf;

   int const numPostRestricted = postLoc->nx * postLoc->ny * postLoc->nf;

   int nbatch = preLoc->nbatch;
   pvAssert(nbatch == postLoc->nbatch);

   const int sy  = postLoc->nx * postLoc->nf; // stride in restricted layer
   const int syw = weights->getGeometry()->getPatchStrideY(); // stride in patch

   bool const preLayerIsSparse = mPreLayer->getSparseFlag();

   int numAxonalArbors = mArborList->getNumAxonalArbors();
   for (int arbor = 0; arbor < numAxonalArbors; arbor++) {
      int delay                = mArborList->getDelay(arbor);
      PVLayerCube activityCube = mPreLayer->getPublisher()->createCube(delay);

      for (int b = 0; b < nbatch; b++) {
         size_t batchOffset                                 = b * numPreExtended;
         float *activityBatch                               = activityCube.data + batchOffset;
         float *gSynPatchHeadBatch                          = postChannel + b * numPostRestricted;
         SparseList<float>::Entry const *activeIndicesBatch = NULL;
         if (preLayerIsSparse) {
            activeIndicesBatch =
                  (SparseList<float>::Entry *)activityCube.activeIndices + batchOffset;
         }

         int numNeurons =
               preLayerIsSparse ? activityCube.numActive[b] : mPreLayer->getNumExtended();

#ifdef PV_USE_OPENMP_THREADS
         clearThreadGSyn();
#endif

         std::size_t const *gSynPatchStart = weights->getGeometry()->getGSynPatchStart().data();
         if (!preLayerIsSparse) {
            for (int y = 0; y < weights->getPatchSizeY(); y++) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(guided)
#endif
               for (int idx = 0; idx < numNeurons; idx++) {
                  int kPreExt = idx;

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

                  float *v                  = postPatchStart + y * sy;
                  float const *weightValues = weightDataStart + y * syw;
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
               for (int idx = 0; idx < numNeurons; idx++) {
                  int kPreExt = activeIndicesBatch[idx].index;

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

                  float *v                  = postPatchStart + y * sy;
                  float const *weightValues = weightDataStart + y * syw;
                  for (int k = 0; k < nk; k++) {
                     v[k] += a * weightValues[k];
                  }
               }
            }
         }
         accumulateThreadGSyn(gSynPatchHeadBatch);
      } // Loop over batch elements
   } // Loop over arbors
#ifdef PV_USE_CUDA
   // CPU updated GSyn, now need to update GSyn on GPU
   mPostLayer->setUpdatedDeviceGSynFlag(true);
#endif // PV_USE_CUDA
}

void PresynapticPerspectiveConvolveDelivery::deliverUnitInput(float *recvBuffer) {
   PVLayerLoc const *postLoc = mPostLayer->getLayerLoc();
   Weights *weights          = mWeightsPair->getPreWeights();

   int const numPostRestricted = postLoc->nx * postLoc->ny * postLoc->nf;

   int nbatch = postLoc->nbatch;

   const int sy  = postLoc->nx * postLoc->nf; // stride in restricted layer
   const int syw = weights->getGeometry()->getPatchStrideY(); // stride in patch

   int numAxonalArbors = mArborList->getNumAxonalArbors();
   for (int arbor = 0; arbor < numAxonalArbors; arbor++) {
      for (int b = 0; b < nbatch; b++) {
         float *recvBatch                                   = recvBuffer + b * numPostRestricted;
         SparseList<float>::Entry const *activeIndicesBatch = NULL;

         int numNeurons = mPreLayer->getNumExtended();

#ifdef PV_USE_OPENMP_THREADS
         clearThreadGSyn();
#endif

         std::size_t const *gSynPatchStart = weights->getGeometry()->getGSynPatchStart().data();
         for (int y = 0; y < weights->getPatchSizeY(); y++) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(guided)
#endif
            for (int idx = 0; idx < numNeurons; idx++) {
               int kPreExt = idx;

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

               float *v                  = postPatchStart + y * sy;
               float const *weightValues = weightDataStart + y * syw;
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
