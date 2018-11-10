/*
 * PresynapticPerspectiveStochasticDelivery.cpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#include "PresynapticPerspectiveStochasticDelivery.hpp"

// Note: there is a lot of code duplication between PresynapticPerspectiveConvolveDelivery
// and PresynapticPerspectiveStochasticDelivery.

namespace PV {

PresynapticPerspectiveStochasticDelivery::PresynapticPerspectiveStochasticDelivery(
      char const *name,
      PVParams *params,
      Communicator *comm) {
   initialize(name, params, comm);
}

PresynapticPerspectiveStochasticDelivery::PresynapticPerspectiveStochasticDelivery() {}

PresynapticPerspectiveStochasticDelivery::~PresynapticPerspectiveStochasticDelivery() {
   delete mRandState;
}

void PresynapticPerspectiveStochasticDelivery::initialize(
      char const *name,
      PVParams *params,
      Communicator *comm) {
   BaseObject::initialize(name, params, comm);
}

void PresynapticPerspectiveStochasticDelivery::setObjectType() {
   mObjectType = "PresynapticPerspectiveStochasticDelivery";
}

int PresynapticPerspectiveStochasticDelivery::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerDelivery::ioParamsFillGroup(ioFlag);
   return status;
}

void PresynapticPerspectiveStochasticDelivery::ioParam_receiveGpu(enum ParamsIOFlag ioFlag) {
   mReceiveGpu = false; // If it's true, we should be using a different class.
}

Response::Status PresynapticPerspectiveStochasticDelivery::communicateInitInfo(
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

Response::Status PresynapticPerspectiveStochasticDelivery::allocateDataStructures() {
   auto status = HyPerDelivery::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   allocateRandState();
#ifdef PV_USE_OPENMP_THREADS
   allocateThreadGSyn();
#endif // PV_USE_OPENMP_THREADS
   return Response::SUCCESS;
}

void PresynapticPerspectiveStochasticDelivery::allocateRandState() {
   mRandState = new Random(mPreLayer->getLayerLoc(), true /*need RNGs in the extended buffer*/);
}

void PresynapticPerspectiveStochasticDelivery::deliver(float *destBuffer) {
   // Check if we need to update based on connection's channel
   if (getChannelCode() == CHANNEL_NOUPDATE) {
      return;
   }
   float *postChannel = destBuffer;
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
         float const *activityBatch                         = activityCube.data + batchOffset;
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
                  taus_uint4 *rng              = mRandState->getRNG(kPreExt);
                  long along                   = (long)((double)a * cl_random_max());

                  float *v                  = postPatchStart + y * sy;
                  float const *weightValues = weightDataStart + y * syw;
                  for (int k = 0; k < nk; k++) {
                     *rng = cl_random_get(*rng);
                     v[k] += (rng->s0 < along) * weightValues[k];
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
                  taus_uint4 *rng              = mRandState->getRNG(kPreExt);
                  long along                   = (long)((double)a * cl_random_max());

                  float *v                  = postPatchStart + y * sy;
                  float const *weightValues = weightDataStart + y * syw;
                  for (int k = 0; k < nk; k++) {
                     *rng = cl_random_get(*rng);
                     v[k] += (rng->s0 < along) * weightValues[k];
                  }
               }
            }
         }
         accumulateThreadGSyn(gSynPatchHeadBatch);
      }
   }
#ifdef PV_USE_CUDA
   // CPU updated GSyn, now need to update GSyn on GPU
   mPostLayer->setUpdatedDeviceGSynFlag(true);
#endif // PV_USE_CUDA
}

void PresynapticPerspectiveStochasticDelivery::deliverUnitInput(float *recvBuffer) {
   PVLayerLoc const *preLoc  = mPreLayer->getLayerLoc();
   PVLayerLoc const *postLoc = mPostLayer->getLayerLoc();
   Weights *weights          = mWeightsPair->getPreWeights();

   int const numPostRestricted = postLoc->nx * postLoc->ny * postLoc->nf;

   int nbatch = postLoc->nbatch;

   const int sy  = postLoc->nx * postLoc->nf; // stride in restricted layer
   const int syw = weights->getGeometry()->getPatchStrideY(); // stride in patch

   int const numAxonalArbors = mArborList->getNumAxonalArbors();
   for (int arbor = 0; arbor < numAxonalArbors; arbor++) {
      int delay                = mArborList->getDelay(arbor);
      PVLayerCube activityCube = mPreLayer->getPublisher()->createCube(delay);

      for (int b = 0; b < nbatch; b++) {
         float *recvBatch = recvBuffer + b * numPostRestricted;
         int numNeurons   = mPreLayer->getNumExtended();

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

               // Activity
               float a = mDeltaTimeFactor;

               float *recvPatchHead = setWorkingGSynBuffer(recvBatch);

               float *postPatchStart = &recvPatchHead[gSynPatchStart[kPreExt]];

               const int nk                 = patch->nx * weights->getPatchSizeF();
               float const *weightDataHead  = weights->getDataFromPatchIndex(arbor, kPreExt);
               float const *weightDataStart = &weightDataHead[patch->offset];
               taus_uint4 *rng              = mRandState->getRNG(kPreExt);
               long along                   = (long)cl_random_max();

               float *v                  = postPatchStart + y * sy;
               float const *weightValues = weightDataStart + y * syw;
               for (int k = 0; k < nk; k++) {
                  *rng = cl_random_get(*rng);
                  v[k] += (rng->s0 < along) * weightValues[k];
               }
            }
         }
#ifdef PV_USE_OPENMP_THREADS
         accumulateThreadGSyn(recvBatch);
#endif // PV_USE_OPENMP_THREADS
      }
   }
}

} // end namespace PV
