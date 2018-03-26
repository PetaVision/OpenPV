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
   allocateThreadGSyn();
   return Response::SUCCESS;
}

void PresynapticPerspectiveConvolveDelivery::allocateThreadGSyn() {
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
         // Clear all thread gsyn buffer
         if (!mThreadGSyn.empty()) {
#pragma omp parallel for schedule(static)
            for (int ti = 0; ti < parent->getNumThreads(); ++ti) {
               for (int ni = 0; ni < numPostRestricted; ++ni) {
                  mThreadGSyn[ti][ni] = 0.0;
               }
            }
         }
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

                  // gSyn
                  float *gSynPatchHead = gSynPatchHeadBatch;

#ifdef PV_USE_OPENMP_THREADS
                  if (!mThreadGSyn.empty()) {
                     gSynPatchHead = mThreadGSyn[omp_get_thread_num()].data();
                  }
#endif // PV_USE_OPENMP_THREADS

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

                  // gSyn
                  float *gSynPatchHead = gSynPatchHeadBatch;

#ifdef PV_USE_OPENMP_THREADS
                  if (!mThreadGSyn.empty()) {
                     gSynPatchHead = mThreadGSyn[omp_get_thread_num()].data();
                  }
#endif // PV_USE_OPENMP_THREADS

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
#ifdef PV_USE_OPENMP_THREADS
         // Accumulate back into gSyn. Should this be done in HyPerLayer where it can be done once,
         // as opposed to once per connection?
         if (!mThreadGSyn.empty()) {
            float *gSynPatchHead = gSynPatchHeadBatch;
            int numNeurons       = mPostLayer->getNumNeurons();
            for (int ti = 0; ti < parent->getNumThreads(); ti++) {
               float *onethread = mThreadGSyn[ti].data();
// Looping over neurons is thread safe
#pragma omp parallel for
               for (int ni = 0; ni < numNeurons; ni++) {
                  gSynPatchHead[ni] += onethread[ni];
               }
            }
         }
#endif // PV_USE_OPENMP_THREADS
      }
   }
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
         // Clear all thread gsyn buffer
         if (!mThreadGSyn.empty()) {
#pragma omp parallel for schedule(static)
            for (int ti = 0; ti < parent->getNumThreads(); ++ti) {
               for (int ni = 0; ni < numPostRestricted; ++ni) {
                  mThreadGSyn[ti][ni] = 0.0;
               }
            }
         }
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

               // gSyn
               float *recvPatchHead = recvBatch;

#ifdef PV_USE_OPENMP_THREADS
               if (!mThreadGSyn.empty()) {
                  recvPatchHead = mThreadGSyn[omp_get_thread_num()].data();
               }
#endif // PV_USE_OPENMP_THREADS

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
#ifdef PV_USE_OPENMP_THREADS
         // Accumulate back into gSyn. Should this be done in HyPerLayer where it can be done once,
         // as opposed to once per connection?
         if (!mThreadGSyn.empty()) {
            float *recvPatchHead = recvBatch;
            int numNeurons       = mPostLayer->getNumNeurons();
            for (int ti = 0; ti < parent->getNumThreads(); ti++) {
               float *onethread = mThreadGSyn[ti].data();
// Looping over neurons is thread safe
#pragma omp parallel for
               for (int ni = 0; ni < numNeurons; ni++) {
                  recvPatchHead[ni] += onethread[ni];
               }
            }
         }
#endif // PV_USE_OPENMP_THREADS
      }
   }
}

} // end namespace PV
