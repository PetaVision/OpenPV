/*
 * PostsynapticPerspectiveGPUDelivery.cpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#include "PostsynapticPerspectiveGPUDelivery.hpp"

namespace PV {

PostsynapticPerspectiveGPUDelivery::PostsynapticPerspectiveGPUDelivery(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   mCorrectReceiveGpu = true;
   initialize(name, params, comm);
}

PostsynapticPerspectiveGPUDelivery::PostsynapticPerspectiveGPUDelivery() {
   mCorrectReceiveGpu = true;
}

PostsynapticPerspectiveGPUDelivery::~PostsynapticPerspectiveGPUDelivery() {
   delete mRecvKernel;
   delete mDevicePostToPreActivity;
   delete mCudnnGSyn;
}

void PostsynapticPerspectiveGPUDelivery::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   mReceiveGpu = true; // If it's false, we should be using a different class.
   BaseObject::initialize(name, params, comm);
}

void PostsynapticPerspectiveGPUDelivery::setObjectType() {
   mObjectType = "PostsynapticPerspectiveGPUDelivery";
}

Response::Status PostsynapticPerspectiveGPUDelivery::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = HyPerDelivery::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   if (getChannelCode() == CHANNEL_NOUPDATE) { return status; }
   // HyPerDelivery::communicateInitInfo() postpones until mWeightsPair communicates.
   pvAssert(mWeightsPair and mWeightsPair->getInitInfoCommunicatedFlag());
   mWeightsPair->needPost();
   // Tell pre and post layers to allocate memory on gpu, which they will do
   // during the AllocateDataStructures stage.

   // we need pre datastore, weights, and post gsyn for the channelCode allocated on the GPU.
   mPreData->setAllocCudaDatastore();
   mWeightsPair->getPostWeights()->useGPU();
   mPostGSyn->useCuda();

   return Response::SUCCESS;
}

Response::Status PostsynapticPerspectiveGPUDelivery::setCudaDevice(
      std::shared_ptr<SetCudaDeviceMessage const> message) {
   auto status = HyPerDelivery::setCudaDevice(message);
   if (status != Response::SUCCESS) {
      return status;
   }
   if (getChannelCode() == CHANNEL_NOUPDATE) { return status; }
   pvAssert(mUsingGPUFlag);
   mWeightsPair->getPostWeights()->setCudaDevice(message->mCudaDevice);
   // Increment number of postKernels for cuDNN workspace memory
   mCudaDevice->incrementConvKernels();
   return status;
}

Response::Status PostsynapticPerspectiveGPUDelivery::allocateDataStructures() {
   FatalIf(
         mCudaDevice == nullptr,
         "%s received AllocateData without having received SetCudaDevice.\n",
         getDescription_c());
   if (mWeightsPair and !mWeightsPair->getDataStructuresAllocatedFlag()) {
      return Response::POSTPONE;
   }
   if (mPostGSyn and !mPostGSyn->getDataStructuresAllocatedFlag()) {
      return Response::POSTPONE;
   }

   auto status = HyPerDelivery::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   if (getChannelCode() == CHANNEL_NOUPDATE) { return status; }

   int const numPostRestricted = mPostGSyn->getBufferSize();
   mDevicePostToPreActivity =
         mCudaDevice->createBuffer(numPostRestricted * sizeof(long), &getDescription());

   // mCudnnGSyn only needs the GSyn channel used by the delivery object
   int const numPostAcrossBatch = mPostGSyn->getBufferSizeAcrossBatch();
   mCudnnGSyn = mCudaDevice->createBuffer(numPostAcrossBatch * sizeof(float), &getDescription());

   return Response::SUCCESS;
}

Response::Status PostsynapticPerspectiveGPUDelivery::copyInitialStateToGPU() {
   initializeRecvKernelArgs();
   return Response::SUCCESS;
}

void PostsynapticPerspectiveGPUDelivery::initializeRecvKernelArgs() {
   if (getChannelCode() == CHANNEL_NOUPDATE) { return; }
   PVCuda::CudaDevice *device = mCudaDevice;
   Weights *postWeights       = mWeightsPair->getPostWeights();
   mRecvKernel                = new PVCuda::CudaRecvPost(device);

   PVLayerLoc const *preLoc  = mPreData->getLayerLoc();
   PVLayerLoc const *postLoc = mPostGSyn->getLayerLoc();

   PVCuda::CudaBuffer *d_PreData           = mPreData->getCudaDatastore();
   PVCuda::CudaBuffer *d_PostGSyn          = mPostGSyn->getCudaBuffer();
   PVCuda::CudaBuffer *d_PatchToDataLookup = postWeights->getDevicePatchToDataLookup();
   PVCuda::CudaBuffer *d_WData             = postWeights->getDeviceData();

#ifdef PV_USE_CUDNN
   PVCuda::CudaBuffer *cudnn_preData = mPreData->getCudnnDatastore();
   PVCuda::CudaBuffer *cudnn_WData   = postWeights->getCUDNNData();
   pvAssert(cudnn_preData);
   pvAssert(mCudnnGSyn);
   pvAssert(cudnn_WData);
#endif

   pvAssert(d_PreData);
   pvAssert(d_PostGSyn);
   pvAssert(d_PatchToDataLookup);
   pvAssert(d_WData);

   int sy           = (preLoc->nx + preLoc->halo.rt + preLoc->halo.lt) * preLoc->nf;
   int syp          = postWeights->getPatchStrideY();
   int numPerStride = postWeights->getPatchSizeX() * postWeights->getPatchSizeF();

   int oNblt = postLoc->halo.lt;
   int oNbrt = postLoc->halo.rt;
   int oNbup = postLoc->halo.up;
   int oNbdn = postLoc->halo.dn;

   // nxp, nyp, and nfp should be orig conn's
   int oNxp   = postWeights->getPatchSizeX();
   int oNyp   = postWeights->getPatchSizeY();
   int oNfp   = postWeights->getPatchSizeF();
   int postNx = postLoc->nx;
   int postNy = postLoc->ny;
   int postNf = postLoc->nf;

   int preNx   = preLoc->nx;
   int preNy   = preLoc->ny;
   int preNf   = preLoc->nf;
   int preNblt = preLoc->halo.lt;
   int preNbrt = preLoc->halo.rt;
   int preNbup = preLoc->halo.up;
   int preNbdn = preLoc->halo.dn;

   int nbatch = preLoc->nbatch;

   // Set local sizes here
   float preToPostScaleX = (float)preLoc->nx / ((float)postLoc->nx);
   float preToPostScaleY = (float)preLoc->ny / ((float)postLoc->ny);

   // The CudaRecvPost kernel needs a buffer containing, for each postsynaptic GSyn neuron,
   // the offset of the start of the receptive field into the presynaptic activity buffer.
   // It expects a buffer of long ints, of length post->getNumNeurons().
   //
   // The relevant information is in the PatchGeometry's mUnshrunkenStart buffer, but this
   // has length post->getNumExtended().
   int const postNumRestricted     = postNx * postNy * postNf;
   auto *h_PostToPreActivityVector = new vector<long>(postNumRestricted);
   auto *h_PostToPreActivity       = h_PostToPreActivityVector->data();
   auto postGeometry               = postWeights->getGeometry();
   for (int k = 0; k < postNumRestricted; k++) {
      int const kExtended = kIndexExtended(k, postNx, postNy, postNf, oNblt, oNbrt, oNbup, oNbdn);
      h_PostToPreActivity[k] = postGeometry->getUnshrunkenStart(kExtended);
   }
   mDevicePostToPreActivity->copyToDevice(h_PostToPreActivity);
   delete h_PostToPreActivityVector;
   h_PostToPreActivityVector = nullptr;
   h_PostToPreActivity       = nullptr;

   // See the size of buffer needed based on x and y
   // oNxp is the patch size from the post point of view

   if (mCommunicator->globalCommRank() == 0) {
      InfoLog() << "preToPostScale: (" << preToPostScaleX << "," << preToPostScaleY << ")\n";
   }

   mRecvKernel->setArgs(
         nbatch,
         postNx, // num post neurons
         postNy,
         postNf,

         oNblt, // Border of orig
         oNbrt, // Border of orig
         oNbdn, // Border of orig
         oNbup, // Border of orig

         preNx,
         preNy,
         preNf,
         preNblt,
         preNbrt,
         preNbup,
         preNbdn,

         oNxp,
         oNyp,
         oNfp,

         preToPostScaleX,
         preToPostScaleY,

         sy,
         syp,
         numPerStride,
         mDeltaTimeFactor,
         postWeights->getSharedFlag(),

         mDevicePostToPreActivity,
         d_PreData,
         d_WData,
         d_PostGSyn,
#ifdef PV_USE_CUDNN
         cudnn_preData,
         cudnn_WData,
         mCudnnGSyn,
#endif
         d_PatchToDataLookup);
}

void PostsynapticPerspectiveGPUDelivery::deliver(float *destBuffer) {
   // Check if we need to update based on connection's channel
   if (getChannelCode() == CHANNEL_NOUPDATE) {
      return;
   }
   pvAssert(mRecvKernel);

   int numAxonalArbors = mArborList->getNumAxonalArbors();
   for (int arbor = 0; arbor < numAxonalArbors; arbor++) {

      mRecvKernel->set_dt_factor(mDeltaTimeFactor);
      // Only copy pre activity to GPU if it's updated on the CPU since last GPU update.
      if (mPreData->getUpdatedCudaDatastoreFlag()) {
         int delay                            = mArborList->getDelay(arbor);
         PVLayerCube activityCube             = mPreData->getPublisher()->createCube(delay);
         float const *h_preDatastore          = activityCube.data;
         PVCuda::CudaBuffer *cudaPreDatastore = mPreData->getCudaDatastore();
         pvAssert(cudaPreDatastore);
         cudaPreDatastore->copyToDevice(h_preDatastore);
         // Device now has updated
         mPreData->setUpdatedCudaDatastoreFlag(false);
      }

      // Permutation buffer is local to the kernel, NOT the layer
      // Therefore, we must permute Datastore every time
      mRecvKernel->permuteDatastorePVToCudnn();
      //}

      // Permute GSyn
      mRecvKernel->permuteGSynPVToCudnn(getChannelCode());

      PVLayerLoc const *postLoc = mPostGSyn->getLayerLoc();
      int totF                  = postLoc->nf;
      int totX                  = postLoc->nx;
      int totY                  = postLoc->ny;
      // Make sure local sizes are divisible by f, x, and y
      mRecvKernel->run(totX, totY, totF, 1L, 1L, 1L);

#ifdef PV_USE_CUDNN
      mRecvKernel->permuteGSynCudnnToPV(getChannelCode());
#endif
   }
}

// This is a copy of PostsynapticPerspectiveConvolveDelivery.
// The spirit of this class says we should put this method on the GPU,
// but the priority for doing so is rather low.
void PostsynapticPerspectiveGPUDelivery::deliverUnitInput(
      float *recvBuffer) {
   // Check if we need to update based on connection's channel
   if (getChannelCode() == CHANNEL_NOUPDATE) {
      return;
   }

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
                     dv += weightValues[k];
                  }
                  *recvLocation += mDeltaTimeFactor * dv;
               }
            }
         }
      }
   }
}

} // end namespace PV
