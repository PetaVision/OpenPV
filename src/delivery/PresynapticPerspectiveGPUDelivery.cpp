/*
 * PresynapticPerspectiveGPUDelivery.cpp
 *
 *  Created on: Jan 10, 2018
 *      Author: Pete Schultz
 */

#include "PresynapticPerspectiveGPUDelivery.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

PresynapticPerspectiveGPUDelivery::PresynapticPerspectiveGPUDelivery(
      char const *name,
      HyPerCol *hc) {
   initialize(name, hc);
}

PresynapticPerspectiveGPUDelivery::PresynapticPerspectiveGPUDelivery() {}

PresynapticPerspectiveGPUDelivery::~PresynapticPerspectiveGPUDelivery() {}

int PresynapticPerspectiveGPUDelivery::initialize(char const *name, HyPerCol *hc) {
   return BaseObject::initialize(name, hc);
}

void PresynapticPerspectiveGPUDelivery::setObjectType() {
   mObjectType = "PresynapticPerspectiveGPUDelivery";
}

int PresynapticPerspectiveGPUDelivery::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerDelivery::ioParamsFillGroup(ioFlag);
   return status;
}

void PresynapticPerspectiveGPUDelivery::ioParam_receiveGpu(enum ParamsIOFlag ioFlag) {
   mReceiveGpu = true; // If it's false, we should be using a different class.
}

Response::Status PresynapticPerspectiveGPUDelivery::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = HyPerDelivery::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   // HyPerDelivery::communicateInitInfo() postpones until mWeightsPair communicates.
   pvAssert(mWeightsPair and mWeightsPair->getInitInfoCommunicatedFlag());
   mWeightsPair->needPre();
   // Tell pre and post layers to allocate memory on gpu, which they will do
   // during the AllocateDataStructures stage.

   // we need pre datastore, weights, and post gsyn for the channelCode allocated on the GPU.
   getPreLayer()->setAllocDeviceDatastore();
   mWeightsPair->getPreWeights()->useGPU();
   getPostLayer()->setAllocDeviceGSyn();

   // If recv from pre and pre layer is sparse, allocate activeIndices
   if (!mUpdateGSynFromPostPerspective && getPreLayer()->getSparseFlag()) {
      getPreLayer()->setAllocDeviceActiveIndices();
   }

   return status;
}

Response::Status PresynapticPerspectiveGPUDelivery::setCudaDevice(
      std::shared_ptr<SetCudaDeviceMessage const> message) {
   pvAssert(mUsingGPUFlag);
   auto status = HyPerDelivery::setCudaDevice(message);
   if (status != Response::SUCCESS) {
      return status;
   }
   mWeightsPair->getPreWeights()->setCudaDevice(message->mCudaDevice);
   mCudaDevice = message->mCudaDevice;
   return status;
}

Response::Status PresynapticPerspectiveGPUDelivery::allocateDataStructures() {
   FatalIf(
         mCudaDevice == nullptr,
         "%s received AllocateData without having received SetCudaDevice.\n",
         getDescription_c());
   if (!mWeightsPair->getDataStructuresAllocatedFlag()) {
      return Response::POSTPONE;
   }

   auto status = HyPerDelivery::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }

   initializeRecvKernelArgs();

   allocateThreadGSyn(); // Needed for deliverUnitInput, because it doesn't use GPU yet.
   return Response::SUCCESS;
}

void PresynapticPerspectiveGPUDelivery::initializeRecvKernelArgs() {
   PVCuda::CudaDevice *device = parent->getDevice();
   Weights *preWeights        = mWeightsPair->getPreWeights();
   mRecvKernel                = new PVCuda::CudaRecvPre(device);

   const PVLayerLoc *preLoc  = getPreLayer()->getLayerLoc();
   const PVLayerLoc *postLoc = getPostLayer()->getLayerLoc();
   const PVHalo *preHalo     = &getPreLayer()->getLayerLoc()->halo;
   const PVHalo *postHalo    = &getPostLayer()->getLayerLoc()->halo;

   PVCuda::CudaBuffer *d_PreData           = getPreLayer()->getDeviceDatastore();
   PVCuda::CudaBuffer *d_PostGSyn          = getPostLayer()->getDeviceGSyn();
   PVCuda::CudaBuffer *d_PatchToDataLookup = preWeights->getDevicePatchToDataLookup();
   PVCuda::CudaBuffer *d_WData             = preWeights->getDeviceData();

   pvAssert(d_PreData);
   pvAssert(d_PostGSyn);
   pvAssert(d_PatchToDataLookup);
   pvAssert(d_WData);

   // We create mDevicePatches and mDeviceGSynPatchStart here, as opposed to creating them in
   // the Weights object, because they are only needed by presynaptic-perspective delivery.
   auto preGeometry = preWeights->getGeometry();
   std::size_t const numPatches = (std::size_t)preGeometry->getNumPatches();
   std::size_t size;

   Patch const *hostPatches = &preGeometry->getPatch(0); // Patches allocated as one vector
   size                     = (std::size_t)numPatches * sizeof(*hostPatches);
   mDevicePatches           = mCudaDevice->createBuffer(size, &description);
   pvAssert(mDevicePatches);
   // Copy patch geometry information onto CUDA device because it never changes.
   mDevicePatches->copyToDevice(hostPatches);

   auto const *hostGSynPatchStart = preGeometry->getGSynPatchStart().data();
   size                           = (std::size_t)numPatches * sizeof(*hostGSynPatchStart);
   mDeviceGSynPatchStart          = mCudaDevice->createBuffer(size, &description);
   // Copy GSynPatchStart array onto CUDA device because it never changes.
   pvAssert(mDeviceGSynPatchStart);
   mDeviceGSynPatchStart->copyToDevice(hostGSynPatchStart);

   int nxp = mWeightsPair->getPreWeights()->getPatchSizeX();
   int nyp = mWeightsPair->getPreWeights()->getPatchSizeY();
   int nfp = mWeightsPair->getPreWeights()->getPatchSizeF();

   int sy  = postLoc->nx * postLoc->nf; // stride in restricted post layer
   int syw = preWeights->getPatchStrideY();

   bool isSparse = getPreLayer()->getSparseFlag();

   int numPreExt  = getPreLayer()->getNumExtended();
   int numPostRes = getPostLayer()->getNumNeurons();

   int nbatch = postLoc->nbatch;

   PVCuda::CudaBuffer *d_activeIndices = NULL;
   PVCuda::CudaBuffer *d_numActive     = NULL;
   if (isSparse) {
      d_numActive = getPreLayer()->getDeviceNumActive();
      pvAssert(d_numActive);
      d_activeIndices = getPreLayer()->getDeviceActiveIndices();
      pvAssert(d_activeIndices);
   }

   mRecvKernel->setArgs(
         nbatch,
         numPreExt,
         numPostRes,
         nxp,
         nyp,
         nfp,

         sy,
         syw,
         mDeltaTimeFactor,
         preWeights->getSharedFlag(),
         mDevicePatches,
         mDeviceGSynPatchStart,

         d_PreData,
         preWeights->getDeviceData(),
         d_PostGSyn,
         d_PatchToDataLookup,

         isSparse,
         d_numActive,
         d_activeIndices);
}

void PresynapticPerspectiveGPUDelivery::allocateThreadGSyn() {
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

void PresynapticPerspectiveGPUDelivery::deliver() {
   // Check if we need to update based on connection's channel
   if (getChannelCode() == CHANNEL_NOUPDATE) {
      return;
   }
   float *postChannel = mPostLayer->getChannel(getChannelCode());
   pvAssert(postChannel);

   pvAssert(mRecvKernel);

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

      mRecvKernel->set_dt_factor(mDeltaTimeFactor);

      // Post layer receives synaptic input
      // Only with respect to post layer
      const PVLayerLoc *preLoc  = getPreLayer()->getLayerLoc();
      const PVLayerLoc *postLoc = getPostLayer()->getLayerLoc();
      // If the connection uses gpu to receive, update all buffers

      // Update pre datastore, post gsyn, and conn weights only if they're updated
      if (getPreLayer()->getUpdatedDeviceDatastoreFlag()) {
         float *h_preDatastore              = activityCube.data;
         PVCuda::CudaBuffer *d_preDatastore = getPreLayer()->getDeviceDatastore();
         pvAssert(d_preDatastore);
         d_preDatastore->copyToDevice(h_preDatastore);

         // Copy active indices and num active if needed
         if (activityCube.isSparse) {
            PVCuda::CudaBuffer *d_ActiveIndices;
            PVCuda::CudaBuffer *d_numActive;
            d_ActiveIndices = getPreLayer()->getDeviceActiveIndices();
            d_numActive     = getPreLayer()->getDeviceNumActive();
            pvAssert(d_ActiveIndices);
            SparseList<float>::Entry const *h_ActiveIndices =
                  (SparseList<float>::Entry *)activityCube.activeIndices;
            long const *h_numActive = activityCube.numActive;
            pvAssert(h_ActiveIndices);
            d_numActive->copyToDevice(h_numActive);
            d_ActiveIndices->copyToDevice(h_ActiveIndices);
         }
         // Device now has updated
         getPreLayer()->setUpdatedDeviceDatastoreFlag(false);
      }

      // X direction is active neuron
      // Y direction is post patch size
      long totActiveNeuron[parent->getNBatch()];
      long maxTotalActiveNeuron = 0;
      for (int b = 0; b < parent->getNBatch(); b++) {
         if (activityCube.isSparse) {
            totActiveNeuron[b] = activityCube.numActive[b];
         }
         else {
            totActiveNeuron[b] = getPreLayer()->getNumExtended();
         }
         if (totActiveNeuron[b] > maxTotalActiveNeuron) {
            maxTotalActiveNeuron = totActiveNeuron[b];
         }
      }

      long totPatchSize   = (long)weights->getPatchSizeOverall();
      long totThreads     = maxTotalActiveNeuron * totPatchSize;
      int maxThreads      = parent->getDevice()->get_max_threads();
      int numLocalThreads = totPatchSize < maxThreads ? totPatchSize : maxThreads;

      mRecvKernel->run_nocheck(totThreads, numLocalThreads);
   }
   // GSyn already living on GPU
   mPostLayer->setUpdatedDeviceGSynFlag(false);
}

// This is a copy of PresynapticPerspectiveConvolveDelivery.
// The spirit of this class says we should put this method on the GPU,
// but the priority for doing so is rather low.
void PresynapticPerspectiveGPUDelivery::deliverUnitInput(float *recvBuffer) {
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
