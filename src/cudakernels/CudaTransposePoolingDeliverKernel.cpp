/*
 * CudaTransposePoolingDeliverKernel.cpp
 *
 *  Created on: Aug 16, 2016
 *      Author: pschultz
 */

#include "cudakernels/CudaTransposePoolingDeliverKernel.hpp"
#include "arch/cuda/cuda_util.hpp"
#include "utils/PVAssert.hpp"
#include <cmath>
#include <vector> // Added for debugging

namespace PVCuda {

CudaTransposePoolingDeliverKernel::CudaTransposePoolingDeliverKernel(CudaDevice* inDevice) : CudaKernel(inDevice) {
   kernelName = "CudaTransposePoolingDeliverKernel";
}

CudaTransposePoolingDeliverKernel::~CudaTransposePoolingDeliverKernel() {
}

void CudaTransposePoolingDeliverKernel::setArgs(
      PVLayerLoc const * preLoc,
      PVLayerLoc const * postLoc,
      PVLayerLoc const * origConnPreLoc,
      PVLayerLoc const * origConnPostLoc,
      int nxpPost,
      int nypPost,
      cudnnPoolingMode_t poolingMode,
      int multiplier,
      CudaBuffer * dataStoreBuffer,
      CudaBuffer * gSynBuffer,
      CudaBuffer * origConnDataStoreBuffer,
      CudaBuffer * origConnGSynBuffer,
      int channel) {
   mPreLoc = preLoc;
   mPostLoc = postLoc;
   pvAssertMessage(preLoc->nx<=postLoc->nx && preLoc->ny<=postLoc->ny, "CudaTransposePoolingDeliverKernel: Transpose pooling requires pre-layer to have same or lower density as post-layer.\n");
   mPoolingMode = poolingMode;
   mMultiplier = (float) multiplier;

   int strideX = CudaPoolingDeliverKernel::calcStride(mPostLoc->nx, mPreLoc->nx);
   int strideY = CudaPoolingDeliverKernel::calcStride(mPostLoc->ny, mPreLoc->ny);
   int nxpPre = nxpPost*mPostLoc->nx/mPreLoc->nx; pvAssert(nxpPre*mPreLoc->nx == nxpPost*mPostLoc->nx);
   int nypPre = nypPost*mPostLoc->ny/mPreLoc->ny; pvAssert(nypPre*mPreLoc->ny == nypPost*mPostLoc->ny);

   cudnnStatus_t status;
   status = cudnnCreatePoolingDescriptor(&mPoolingDescriptor);
   cudnnHandleError(status, "Create pooling descriptor");
#if CUDNN_MAJOR == 5
   status = cudnnSetPooling2dDescriptor(
         mPoolingDescriptor,
         poolingMode,
				 CUDNN_NOT_PROPAGATE_NAN,
         nypPre,
         nxpPre,
         0/*horizontal padding*/,
         0/*vertical padding*/,
         strideY,
         strideX
   );
#elif CUDNN_MAJOR == 4
	 status = cudnnSetPooling2dDescriptor(
         mPoolingDescriptor,
         poolingMode,
         nypPre,
         nxpPre,
         0/*horizontal padding*/,
         0/*vertical padding*/,
         strideY,
         strideX
   );
#else
#error The cuDNN version is required to be either v4 or v5.
#endif

   const PVHalo* preHalo = &mPreLoc->halo;
   mBorderExcessX = calcBorderExcess(mPreLoc->nx, mPostLoc->nx, preHalo->lt, nxpPost);
   mBorderExcessY = calcBorderExcess(mPreLoc->ny, mPostLoc->ny, preHalo->up, nypPost);
   status = cudnnCreateTensorDescriptor(&mDataStoreDescriptor);
   cudnnHandleError(status, "Create input tensor descriptor");
   status = cudnnSetTensor4dDescriptor(
         mDataStoreDescriptor,
         CUDNN_TENSOR_NCHW, // PetaVision arrays are ordered NHWC; they will be permuted to NCHW inside do_run()
         CUDNN_DATA_FLOAT,
         mPreLoc->nbatch, //Number of images
         mPreLoc->nf, //Number of feature maps per image
         mPreLoc->ny + preHalo->up + preHalo->dn - 2*mBorderExcessY, //Height of each feature map
         mPreLoc->nx + preHalo->lt + preHalo->rt - 2*mBorderExcessX
         ); //Width of each feature map
   cudnnHandleError(status, "Set input tensor descriptor");
   mDataStore = (float*) dataStoreBuffer->getPointer();
   std::string str(kernelName);
   mCudnnDataStore = device->createBuffer(dataStoreBuffer->getSize(), &str);

   status = cudnnCreateTensorDescriptor(&mGSynDescriptor);
   cudnnHandleError(status, "Create input tensor descriptor");
   status = cudnnSetTensor4dDescriptor(
         mGSynDescriptor,
         CUDNN_TENSOR_NCHW, // PetaVision arrays are ordered NHWC; they will be permuted to NCHW inside do_run()
         CUDNN_DATA_FLOAT,
         mPreLoc->nbatch, //Number of images
         mPostLoc->nf, //Number of feature maps per image
         mPostLoc->ny, //ny restricted
         mPostLoc->nx
   ); //nx restricted
   cudnnHandleError(status, "Set output tensor descriptor");
   int numGSynNeuronsAcrossBatch = mPostLoc->nx*mPostLoc->ny*mPostLoc->nf*mPostLoc->nbatch;
   float * gSynHead = (float*) gSynBuffer->getPointer();
   mGSyn = &gSynHead[channel*numGSynNeuronsAcrossBatch];
   mCudnnGSyn = device->createBuffer(numGSynNeuronsAcrossBatch*sizeof(float), &str);

   mOrigConnPreLoc = origConnPreLoc;
   mOrigConnPostLoc = origConnPostLoc;

   const PVHalo* origConnPreHalo = &mOrigConnPreLoc->halo;
   mOrigConnBorderExcessX = calcBorderExcess(mOrigConnPreLoc->nx, mOrigConnPostLoc->nx, origConnPreHalo->lt, nxpPost);
   mOrigConnBorderExcessY = calcBorderExcess(mOrigConnPreLoc->ny, mOrigConnPostLoc->ny, origConnPreHalo->up, nypPost);
   status = cudnnCreateTensorDescriptor(&mOrigConnDataStoreDescriptor);
   cudnnHandleError(status, "Create original conn pre datastore tensor descriptor");
   status = cudnnSetTensor4dDescriptor(
         mOrigConnDataStoreDescriptor,
         CUDNN_TENSOR_NCHW, // PetaVision arrays are ordered NHWC; they will be permuted to NCHW inside do_run()
         CUDNN_DATA_FLOAT,
         mOrigConnPreLoc->nbatch, //Number of images
         mOrigConnPreLoc->nf, //Number of feature maps per image
         mOrigConnPreLoc->ny + origConnPreHalo->up + origConnPreHalo->dn - 2*mOrigConnBorderExcessY, //Height of each feature map
         mOrigConnPreLoc->nx + origConnPreHalo->lt + origConnPreHalo->rt - 2*mOrigConnBorderExcessX
         ); //Width of each feature map
   cudnnHandleError(status, "Set original conn pre datastore tensor descriptor");
   mOrigConnDataStore = (float*) origConnDataStoreBuffer->getPointer();
   mCudnnOrigConnDataStore = device->createBuffer(origConnDataStoreBuffer->getSize(), &str);

   status = cudnnCreateTensorDescriptor(&mOrigConnGSynDescriptor);
   cudnnHandleError(status, "Create original conn post gsyn tensor descriptor");
   status = cudnnSetTensor4dDescriptor(
         mOrigConnGSynDescriptor,
         CUDNN_TENSOR_NCHW, // PetaVision arrays are ordered NHWC; they will be permuted to NCHW inside do_run()
         CUDNN_DATA_FLOAT,
         mOrigConnPostLoc->nbatch, //Number of images
         mOrigConnPostLoc->nf, //Number of feature maps per image
         mOrigConnPostLoc->ny, //ny restricted
         mOrigConnPostLoc->nx
   ); //nx restricted
   cudnnHandleError(status, "Set original conn post gsyn tensor descriptor");
   int numOrigConnGSynNeuronsAcrossBatch = mOrigConnPostLoc->nf*mOrigConnPostLoc->ny*mOrigConnPostLoc->nf*mOrigConnPostLoc->nbatch;
   float * origConnGSynHead = (float*) origConnGSynBuffer->getPointer();
   mOrigConnGSyn = &origConnGSynHead[channel*numOrigConnGSynNeuronsAcrossBatch];
   mCudnnOrigConnGSyn = device->createBuffer(numOrigConnGSynNeuronsAcrossBatch*sizeof(float), &str);
}

int CudaTransposePoolingDeliverKernel::calcBorderExcess(int preRestricted, int postRestricted, int border, int patchSizePostPerspective) {
   int borderNeeded = (patchSizePostPerspective-1)/2;
   return border - borderNeeded;
}

int CudaTransposePoolingDeliverKernel::calcManyScale(int preRestricted, int postRestricted) {
   int manyScale = postRestricted/preRestricted;
   if (manyScale*preRestricted != postRestricted) { throw; }
   return manyScale;
}

int CudaTransposePoolingDeliverKernel::calcStride(int preRestricted, int postRestricted) {
   return 1;
}

int CudaTransposePoolingDeliverKernel::do_run() {
   float scalingFactor = 1.0f;

   int const blockSize = device->get_max_threads();

   // Permute PV-organized DataStore to CUDNN organization.
   PVHalo const * halo = &mPreLoc->halo;
   int const nxPreExt = mPreLoc->nx + halo->lt + halo->rt;
   int const nyPreExt = mPreLoc->ny + halo->dn + halo->up;
   int const nf = mPreLoc->nf;
   int const nbatch = mPreLoc->nbatch;
   //Calculate grid and work size
   int numNeurons = nbatch * nyPreExt * nxPreExt * nf;
   //Ceil to get all neurons
   int const gridSizePre = std::ceil((float)numNeurons/blockSize);
   float * cudnnDataStorePointer = (float*) mCudnnDataStore->getPointer();
   callPermuteDatastorePVToCudnnKernel(gridSizePre, blockSize, mDataStore, cudnnDataStorePointer, nbatch, nyPreExt, nxPreExt, nf, mBorderExcessX, mBorderExcessY);
   handleCallError("CudaTransposeConn: permute DataStore PV to CUDNN");

   // Permute the PV-ordered GSyn channel to CUDNN ordering.
   int const nxPost = mPostLoc->nx;
   int const nyPost = mPostLoc->ny;
   pvAssert(nf==mPostLoc->nf);
   pvAssert(mPostLoc->nbatch == mPreLoc->nbatch);
   //Calculate grid and work size
   numNeurons = nbatch * nxPost * nyPost * nf;
   float * cudnnGSynPointer = (float*) mCudnnGSyn->getPointer();
   //Ceil to get all neurons
   int const gridSizePost = std::ceil((float)numNeurons/(float)blockSize);
   callPermuteGSynPVToCudnnKernel(gridSizePost, blockSize, mGSyn, cudnnGSynPointer, nbatch, nyPost, nxPost, nf, 1, 1);
   handleCallError("CudaTransposeConn: permute GSyn PV to CUDNN");

   // Permute PV-organized original conn's DataStore to CUDNN organization.
   PVHalo const * origConnHalo = &mOrigConnPreLoc->halo;
   int const origConnNxPreExt = mOrigConnPreLoc->nx + origConnHalo->lt + origConnHalo->rt;
   int const origConnNyPreExt = mOrigConnPreLoc->ny + origConnHalo->dn + origConnHalo->up;
   pvAssert(nf==mOrigConnPreLoc->nf);
   pvAssert(nbatch==mOrigConnPreLoc->nbatch);
   //Calculate grid and work size
   numNeurons = nbatch * origConnNyPreExt * origConnNxPreExt * nf;
   //Ceil to get all neurons
   int const gridSizeOrigConnPre = std::ceil((float)numNeurons/blockSize);
   float * cudnnOrigConnDataStorePointer = (float*) mCudnnOrigConnDataStore->getPointer();
   callPermuteDatastorePVToCudnnKernel(gridSizeOrigConnPre, blockSize, mOrigConnDataStore, cudnnOrigConnDataStorePointer, nbatch, origConnNyPreExt, origConnNxPreExt, nf, mBorderExcessX, mBorderExcessY);
   handleCallError("CudaTransposeConn: permute original conn's DataStore PV to CUDNN");

   // Permute the PV-ordered original conn's GSyn channel to CUDNN ordering.
   int const origConnNxPost = mOrigConnPostLoc->nx;
   int const origConnNyPost = mOrigConnPostLoc->ny;
   pvAssert(nf==mOrigConnPostLoc->nf);
   pvAssert(mOrigConnPostLoc->nbatch==nbatch);
   //Calculate grid and work size
   numNeurons = nbatch * origConnNxPost * origConnNyPost * nf;
   float * cudnnOrigConnGSynPointer = (float*) mCudnnOrigConnGSyn->getPointer();
   //Ceil to get all neurons
   int const gridSizeOrigConnPost = std::ceil((float)numNeurons/(float)blockSize);
   callPermuteGSynPVToCudnnKernel(gridSizeOrigConnPost, blockSize, mOrigConnGSyn, cudnnOrigConnGSynPointer, nbatch, origConnNyPost, origConnNxPost, nf, 1, 1);
   handleCallError("CudaTransposeConn: permute original conn's GSyn PV to CUDNN");

   // Do the pooling
   cudnnStatus_t status = cudnnPoolingBackward(
         (cudnnHandle_t) device->getCudnnHandle(),
         mPoolingDescriptor,
         &mMultiplier,
         mOrigConnGSynDescriptor, cudnnOrigConnGSynPointer,
         mDataStoreDescriptor, cudnnDataStorePointer,
         mOrigConnDataStoreDescriptor, cudnnOrigConnDataStorePointer,
         &scalingFactor,
         mGSynDescriptor, cudnnGSynPointer
   );
   cudnnHandleError(status, "CudaTransposeConn: backward pooling run");

   device->syncDevice();

   // Permute the CUDNN-ordering GSyn back to PV ordering
   callPermuteGSynCudnnToPVKernel(gridSizePost, blockSize, mGSyn, cudnnGSynPointer, nbatch, nyPost, nxPost, nf, 1, 1);
   handleCallError("CudaTransposeConn: permute GSyn CUDNN back to PV");
   return 0;
}

} /* namespace PVCuda */
