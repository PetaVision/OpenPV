/*
 * PresynapticPerspectiveGPUDelivery.cpp
 *
 *  Created on: Jan 10, 2018
 *      Author: Pete Schultz
 */

#include "PresynapticPerspectiveGPUDelivery.hpp"

namespace PV {

PresynapticPerspectiveGPUDelivery::PresynapticPerspectiveGPUDelivery(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   mCorrectReceiveGpu = true;
   initialize(name, params, comm);
}

PresynapticPerspectiveGPUDelivery::PresynapticPerspectiveGPUDelivery() {
   mCorrectReceiveGpu = true;
}

PresynapticPerspectiveGPUDelivery::~PresynapticPerspectiveGPUDelivery() {}

void PresynapticPerspectiveGPUDelivery::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   mReceiveGpu = true; // If it's false, we should be using a different class.
   BaseObject::initialize(name, params, comm);
}

void PresynapticPerspectiveGPUDelivery::setObjectType() {
   mObjectType = "PresynapticPerspectiveGPUDelivery";
}

Response::Status PresynapticPerspectiveGPUDelivery::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = HyPerDelivery::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   if (getChannelCode() == CHANNEL_NOUPDATE) { return status; }
   // HyPerDelivery::communicateInitInfo() postpones until mWeightsPair communicates.
   pvAssert(mWeightsPair and mWeightsPair->getInitInfoCommunicatedFlag());
   mWeightsPair->needPre();
   // Tell pre and post layers to allocate memory on gpu, which they will do
   // during the AllocateDataStructures stage.

   // we need pre datastore, weights, and post gsyn for the channelCode allocated on the GPU.
   mPreData->setAllocCudaDatastore();
   mWeightsPair->getPreWeights()->useGPU();
   mPostGSyn->useCuda();

   // If recv from pre and pre layer is sparse, allocate activeIndices
   if (mPreData->getSparseLayer()) {
      mPreData->setAllocCudaActiveIndices();
   }

   return status;
}

Response::Status PresynapticPerspectiveGPUDelivery::setCudaDevice(
      std::shared_ptr<SetCudaDeviceMessage const> message) {
   auto status = HyPerDelivery::setCudaDevice(message);
   if (status != Response::SUCCESS) {
      return status;
   }
   if (getChannelCode() == CHANNEL_NOUPDATE) { return status; }
   pvAssert(mUsingGPUFlag);
   mWeightsPair->getPreWeights()->setCudaDevice(message->mCudaDevice);
   mCudaDevice = message->mCudaDevice;
   return status;
}

Response::Status PresynapticPerspectiveGPUDelivery::allocateDataStructures() {
   FatalIf(
         mCudaDevice == nullptr,
         "%s received AllocateData without having received SetCudaDevice.\n",
         getDescription_c());
   if (mWeightsPair and !mWeightsPair->getDataStructuresAllocatedFlag()) {
      return Response::POSTPONE;
   }

   auto status = HyPerDelivery::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   if (getChannelCode() == CHANNEL_NOUPDATE) { return status; }

   // We create mDevicePatches and mDeviceGSynPatchStart here, as opposed to creating them in
   // the Weights object, because they are only needed by presynaptic-perspective delivery.
   auto preGeometry             = mWeightsPair->getPreWeights()->getGeometry();
   std::size_t const numPatches = (std::size_t)preGeometry->getNumPatches();
   std::size_t cudaBufferSize;

   auto const *hostPatches = &preGeometry->getPatch(0); // Only used to get size for allocation
   cudaBufferSize          = (std::size_t)numPatches * sizeof(*hostPatches);
   mDevicePatches          = mCudaDevice->createBuffer(cudaBufferSize, &getDescription());
   pvAssert(mDevicePatches);

   auto const *hostGSynPatchStart = preGeometry->getGSynPatchStart().data();
   cudaBufferSize                 = (std::size_t)numPatches * sizeof(*hostGSynPatchStart);
   mDeviceGSynPatchStart          = mCudaDevice->createBuffer(cudaBufferSize, &getDescription());
   pvAssert(mDeviceGSynPatchStart);

#ifdef PV_USE_OPENMP_THREADS
   allocateThreadGSyn();
#endif // PV_USE_OPENMP_THREADS

   return Response::SUCCESS;
}

Response::Status PresynapticPerspectiveGPUDelivery::copyInitialStateToGPU() {
   initializeRecvKernelArgs();
   return Response::SUCCESS;
}

void PresynapticPerspectiveGPUDelivery::initializeRecvKernelArgs() {
   if (getChannelCode() == CHANNEL_NOUPDATE) { return; }
   PVCuda::CudaDevice *device = mCudaDevice;
   Weights *preWeights        = mWeightsPair->getPreWeights();
   mRecvKernel                = new PVCuda::CudaRecvPre(device);

   const PVLayerLoc *preLoc  = mPreData->getLayerLoc();
   const PVLayerLoc *postLoc = mPostGSyn->getLayerLoc();

   PVCuda::CudaBuffer *d_PreData           = mPreData->getCudaDatastore();
   PVCuda::CudaBuffer *d_PostGSyn          = mPostGSyn->getCudaBuffer();
   PVCuda::CudaBuffer *d_PatchToDataLookup = preWeights->getDevicePatchToDataLookup();
   PVCuda::CudaBuffer *d_WData             = preWeights->getDeviceData();

   pvAssert(d_PreData);
   pvAssert(d_PostGSyn);
   pvAssert(d_PatchToDataLookup);
   pvAssert(d_WData);

   // Copy patch geometry and GSynPatchStart information onto CUDA device
   auto preGeometry         = preWeights->getGeometry();
   Patch const *hostPatches = &preGeometry->getPatch(0); // Patches were allocated as one vector
   mDevicePatches->copyToDevice(hostPatches);

   auto const *hostGSynPatchStart = preGeometry->getGSynPatchStart().data();
   mDeviceGSynPatchStart->copyToDevice(hostGSynPatchStart);

   int nxp = mWeightsPair->getPreWeights()->getPatchSizeX();
   int nyp = mWeightsPair->getPreWeights()->getPatchSizeY();
   int nfp = mWeightsPair->getPreWeights()->getPatchSizeF();

   int sy  = postLoc->nx * postLoc->nf; // stride in restricted post layer
   int syw = preWeights->getPatchStrideY();

   bool isSparse = mPreData->getSparseLayer();

   int const nxPreExt = preLoc->nx + preLoc->halo.lt + preLoc->halo.rt;
   int const nyPreExt = preLoc->ny + preLoc->halo.dn + preLoc->halo.up;
   int numPreExt      = nxPreExt * nyPreExt * preLoc->nf;
   int numPostRes     = mPostGSyn->getBufferSize();

   int nbatch = postLoc->nbatch;

   PVCuda::CudaBuffer *d_activeIndices = NULL;
   PVCuda::CudaBuffer *d_numActive     = NULL;
   if (isSparse) {
      d_numActive = mPreData->getCudaNumActive();
      pvAssert(d_numActive);
      d_activeIndices = mPreData->getCudaActiveIndices();
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
         mChannelCode,
         mDevicePatches,
         mDeviceGSynPatchStart,

         d_PreData,
         d_WData,
         d_PostGSyn,
         d_PatchToDataLookup,

         isSparse,
         d_numActive,
         d_activeIndices);
}

void PresynapticPerspectiveGPUDelivery::deliver(float *destBuffer) {
   // Check if we need to update based on connection's channel
   if (getChannelCode() == CHANNEL_NOUPDATE) {
      return;
   }
   pvAssert(destBuffer);

   pvAssert(mRecvKernel);

   Weights *weights = mWeightsPair->getPreWeights();

   int numAxonalArbors = mArborList->getNumAxonalArbors();
   for (int arbor = 0; arbor < numAxonalArbors; arbor++) {
      int delay                = mArborList->getDelay(arbor);
      PVLayerCube activityCube = mPreData->getPublisher()->createCube(delay);

      mRecvKernel->set_dt_factor(mDeltaTimeFactor);

      PVLayerLoc const *preLoc = &activityCube.loc;
      // If the connection uses gpu to receive, update all buffers

      // Update pre datastore, post gsyn, and conn weights only if they're updated
      if (mPreData->getUpdatedCudaDatastoreFlag()) {
         float const *h_preDatastore        = activityCube.data;
         PVCuda::CudaBuffer *d_preDatastore = mPreData->getCudaDatastore();
         pvAssert(d_preDatastore);
         d_preDatastore->copyToDevice(h_preDatastore);

         // Copy active indices and num active if needed
         if (activityCube.isSparse) {
            PVCuda::CudaBuffer *d_activeIndices;
            PVCuda::CudaBuffer *d_numActive;
            d_activeIndices = mPreData->getCudaActiveIndices();
            d_numActive     = mPreData->getCudaNumActive();
            pvAssert(d_activeIndices);
            SparseList<float>::Entry const *h_ActiveIndices =
                  (SparseList<float>::Entry *)activityCube.activeIndices;
            long const *h_numActive = activityCube.numActive;
            pvAssert(h_ActiveIndices);
            d_numActive->copyToDevice(h_numActive);
            d_activeIndices->copyToDevice(h_ActiveIndices);
         }
         // Device now has updated
         mPreData->setUpdatedCudaDatastoreFlag(false);
      }

      // X direction is active neuron
      // Y direction is post patch size
      int const nbatch = preLoc->nbatch;
      long totActiveNeuron[nbatch];
      long maxTotalActiveNeuron = 0;
      for (int b = 0; b < nbatch; b++) {
         if (activityCube.isSparse) {
            totActiveNeuron[b] = activityCube.numActive[b];
         }
         else {
            totActiveNeuron[b] = activityCube.numItems / activityCube.loc.nbatch;
         }
         if (totActiveNeuron[b] > maxTotalActiveNeuron) {
            maxTotalActiveNeuron = totActiveNeuron[b];
         }
      }

      if (maxTotalActiveNeuron > 0) {
         long totPatchSize   = (long)weights->getPatchSizeOverall();
         long totThreads     = maxTotalActiveNeuron * totPatchSize;
         int maxThreads      = mCudaDevice->get_max_threads();
         int numLocalThreads = totPatchSize < maxThreads ? totPatchSize : maxThreads;

         mRecvKernel->run_nocheck(totThreads, numLocalThreads);
      }
   }
}

// This is a copy of PresynapticPerspectiveConvolveDelivery.
// The spirit of this class says we should put this method on the GPU,
// but the priority for doing so is rather low.
void PresynapticPerspectiveGPUDelivery::deliverUnitInput(float *recvBuffer) {
   PVLayerLoc const *preLoc = mPreData->getLayerLoc();
   int const nxPreExt       = preLoc->nx + preLoc->halo.lt + preLoc->halo.rt;
   int const nyPreExt       = preLoc->ny + preLoc->halo.dn + preLoc->halo.up;
   int const numPreExt      = nxPreExt * nyPreExt * preLoc->nf;

   PVLayerLoc const *postLoc   = mPostGSyn->getLayerLoc();
   int const numPostRestricted = postLoc->nx * postLoc->ny * postLoc->nf;
   int nbatch                  = postLoc->nbatch;
   int const sy                = postLoc->nx * postLoc->nf; // stride in restricted layer

   Weights *weights = mWeightsPair->getPreWeights();
   int const syw    = weights->getGeometry()->getPatchStrideY(); // stride in patch

   int numAxonalArbors = mArborList->getNumAxonalArbors();
   for (int arbor = 0; arbor < numAxonalArbors; arbor++) {
      for (int b = 0; b < nbatch; b++) {
         float *recvBatch = recvBuffer + b * numPostRestricted;

#ifdef PV_USE_OPENMP_THREADS
         clearThreadGSyn();
#endif

         std::size_t const *gSynPatchStart = weights->getGeometry()->getGSynPatchStart().data();
         for (int y = 0; y < weights->getPatchSizeY(); y++) {
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(guided)
#endif
            for (int idx = 0; idx < numPreExt; idx++) {
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
      }
   }
}

} // end namespace PV
