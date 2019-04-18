/*
 * CudaPoolingDeliverKernel.cpp
 *
 *  Created on: Aug 2, 2016
 *      Author: pschultz
 */

#include "arch/cuda/cuda_util.hpp"
#include "utils/PVAssert.hpp"
#include <cmath>
#include <cudakernels/CudaPoolingDeliverKernel.hpp>
#include <cudnn.h>

namespace PVCuda {

CudaPoolingDeliverKernel::CudaPoolingDeliverKernel(CudaDevice *inDevice) : CudaKernel(inDevice) {
   kernelName = "CudaPoolingDeliverKernel";
}

CudaPoolingDeliverKernel::~CudaPoolingDeliverKernel() {
   cudnnDestroyPoolingDescriptor(mPoolingDescriptor);
   cudnnDestroyTensorDescriptor((cudnnTensorDescriptor_t)mDataStoreDescriptor);
   cudnnDestroyTensorDescriptor((cudnnTensorDescriptor_t)mGSynDescriptor);
}

void CudaPoolingDeliverKernel::setArgs(
      PVLayerLoc const *preLoc,
      PVLayerLoc const *postLoc,
      int nxpPost,
      int nypPost,
      cudnnPoolingMode_t poolingMode,
      int multiplier,
      // TODO: instead of passing poolingMode and multiplier, I would prefer
      // to pass the PoolingConn poolingType, and have the setArgs method
      // determine the pooling mode and the multiplier from poolingType and
      // the patch size.  However, this causes a circular dependency between
      // PoolingConn and CudaPoolingRecvPost.  It could be moved into
      // pv_types.h, but it would be nice to define a pooling-specific enum
      // in a pooling-specific file.
      CudaBuffer *dataStoreBuffer,
      CudaBuffer *gSynBuffer,
      int channel) {

   FatalIf(
         preLoc->nx < postLoc->nx,
         "Pooling is not defined for one-to-many connections (pre->nx=%d, post->nx=%d\n",
         preLoc->nx,
         postLoc->nx);
   FatalIf(
         preLoc->ny < postLoc->ny,
         "Pooling is not defined for one-to-many connections (pre->ny=%d, post->ny=%d\n",
         preLoc->ny,
         postLoc->ny);

   mPreLoc      = preLoc;
   mPostLoc     = postLoc;
   mPoolingMode = poolingMode;
   mMultiplier  = (float)multiplier;

   int strideX = calcStride(preLoc->nx, postLoc->nx);
   int strideY = calcStride(preLoc->ny, postLoc->ny);

   cudnnStatus_t status;
   status = cudnnCreatePoolingDescriptor(&mPoolingDescriptor);
   cudnnHandleError(status, "Create pooling descriptor");
   status = cudnnSetPooling2dDescriptor(
         mPoolingDescriptor,
         poolingMode,
#if CUDNN_MAJOR >= 5
         CUDNN_NOT_PROPAGATE_NAN,
#endif
         nypPost,
         nxpPost,
         0 /*horizontal padding*/,
         0 /*vertical padding*/,
         strideY,
         strideX);

   const PVHalo *preHalo = &preLoc->halo;
   mBorderExcessX        = calcBorderExcess(preLoc->nx, postLoc->nx, preHalo->lt, nxpPost);
   mBorderExcessY        = calcBorderExcess(preLoc->ny, postLoc->ny, preHalo->up, nypPost);
   status                = cudnnCreateTensorDescriptor(&mDataStoreDescriptor);
   cudnnHandleError(status, "Create input tensor descriptor");
   status = cudnnSetTensor4dDescriptor(
         mDataStoreDescriptor,
         CUDNN_TENSOR_NCHW, // PetaVision arrays are ordered NHWC; they will be permuted to NCHW
         // inside do_run()
         CUDNN_DATA_FLOAT,
         preLoc->nbatch, // Number of images
         preLoc->nf, // Number of feature maps per image
         preLoc->ny + preHalo->up + preHalo->dn - 2 * mBorderExcessY, // Height of each feature map
         preLoc->nx + preHalo->lt + preHalo->rt - 2 * mBorderExcessX); // Width of each feature map
   mDataStore = (float *)dataStoreBuffer->getPointer();

   status = cudnnCreateTensorDescriptor(&mGSynDescriptor);
   cudnnHandleError(status, "Create input tensor descriptor");
   status = cudnnSetTensor4dDescriptor(
         mGSynDescriptor,
         CUDNN_TENSOR_NCHW, // PetaVision arrays are ordered NHWC; they will be permuted to NCHW
         // inside do_run()
         CUDNN_DATA_FLOAT,
         preLoc->nbatch, // Number of images
         postLoc->nf, // Number of feature maps per image
         postLoc->ny, // ny restricted
         postLoc->nx); // nx restricted
   cudnnHandleError(status, "Set output tensor descriptor");

   std::string str(kernelName);
   mCudnnDataStore = device->createBuffer(dataStoreBuffer->getSize(), &str);

   int numGSynNeuronsAcrossBatch = postLoc->nf * postLoc->ny * postLoc->nf * postLoc->nbatch;
   float *gSynHead               = (float *)gSynBuffer->getPointer();
   mGSyn                         = &gSynHead[channel * numGSynNeuronsAcrossBatch];

   size_t gSynSize = gSynBuffer->getSize();
   mCudnnGSyn      = device->createBuffer(numGSynNeuronsAcrossBatch, &str);
}

int CudaPoolingDeliverKernel::calcBorderExcess(
      int preRestricted,
      int postRestricted,
      int border,
      int patchSizePostPerspective) {
   int preToPostScale = preRestricted / postRestricted;
   int borderNeeded   = (patchSizePostPerspective - preToPostScale) / 2;
   return border - borderNeeded;
}

int CudaPoolingDeliverKernel::calcManyScale(int preRestricted, int postRestricted) { return 1; }

int CudaPoolingDeliverKernel::calcStride(int preRestricted, int postRestricted) {
   return preRestricted / postRestricted;
}

int CudaPoolingDeliverKernel::do_run() {
   float scalingFactor = 1.0f;

   int const blockSize = device->get_max_threads();

   // Permute PV-organized DataStore to CUDNN organization.
   PVHalo const *halo = &mPreLoc->halo;
   int const nxPreExt = mPreLoc->nx + halo->lt + halo->rt;
   int const nyPreExt = mPreLoc->ny + halo->dn + halo->up;
   int const nf       = mPreLoc->nf;
   int const nbatch   = mPreLoc->nbatch;
   // Calculate grid and work size
   int numNeurons = nbatch * nyPreExt * nxPreExt * nf;
   // Ceil to get all neurons
   int const gridSizePre        = std::ceil((float)numNeurons / blockSize);
   float *cudnnDataStorePointer = (float *)mCudnnDataStore->getPointer();
   callPermuteDatastorePVToCudnnKernel(
         gridSizePre,
         blockSize,
         mDataStore,
         cudnnDataStorePointer,
         nbatch,
         nyPreExt,
         nxPreExt,
         nf,
         mBorderExcessX,
         mBorderExcessY);
   handleCallError("Permute DataStore PV to CUDNN");

   // Permute the PV-ordered GSyn channel to CUDNN ordering.
   int const nxPost = mPostLoc->nx;
   int const nyPost = mPostLoc->ny;
   int const nfPost = mPostLoc->nf;
   pvAssert(mPostLoc->nbatch == mPreLoc->nbatch);
   // Calculate grid and work size
   numNeurons              = nbatch * nxPost * nyPost * nf;
   float *cudnnGSynPointer = (float *)mCudnnGSyn->getPointer();
   // Ceil to get all neurons
   int const gridSizePost = std::ceil((float)numNeurons / (float)blockSize);
   callPermuteGSynPVToCudnnKernel(
         gridSizePost, blockSize, mGSyn, cudnnGSynPointer, nbatch, nyPost, nxPost, nf, 1, 1);
   handleCallError("Permute GSyn PV to CUDNN");

   cudnnPoolingMode_t checkMode;
   int h, w, vPad, hPad, vStride, hStride;
#if CUDNN_MAJOR >= 5
   cudnnNanPropagation_t cudnnNanPropagation;
   cudnnGetPooling2dDescriptor(
         (cudnnPoolingDescriptor_t)mPoolingDescriptor,
         &checkMode,
         &cudnnNanPropagation,
         &h,
         &w,
         &vPad,
         &hPad,
         &vStride,
         &hStride);

#elif CUDNN_MAJOR == 4
   cudnnGetPooling2dDescriptor(
         (cudnnPoolingDescriptor_t)mPoolingDescriptor,
         &checkMode,
         &h,
         &w,
         &vPad,
         &hPad,
         &vStride,
         &hStride);
#else
#error The cuDNN version is required to be either v4 or v5.
#endif

   // Do the pooling
   cudnnStatus_t status = cudnnPoolingForward(
         (cudnnHandle_t)device->getCudnnHandle(),
         mPoolingDescriptor,
         &mMultiplier,
         mDataStoreDescriptor,
         cudnnDataStorePointer,
         &scalingFactor,
         mGSynDescriptor,
         cudnnGSynPointer);
   cudnnHandleError(status, "Forward pooling run");

   // Permute the CUDNN-ordering GSyn back to PV ordering
   callPermuteGSynCudnnToPVKernel(
         gridSizePost, blockSize, mGSyn, cudnnGSynPointer, nbatch, nyPost, nxPost, nf, 1, 1);
   handleCallError("Permute GSyn CUDNN back to PV");
   return 0;
}

} /* namespace PVCuda */
