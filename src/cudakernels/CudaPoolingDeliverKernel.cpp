/*
 * CudaPoolingDeliverKernel.cpp
 *
 *  Created on: Aug 2, 2016
 *      Author: pschultz
 */

#include <cudakernels/CudaPoolingDeliverKernel.hpp>
#include "utils/PVAssert.hpp"
#include "arch/cuda/cuda_util.hpp"
#include <cudnn.h>
#include <cmath>

namespace PVCuda {

CudaPoolingDeliverKernel::CudaPoolingDeliverKernel(CudaDevice* inDevice) : CudaKernel(inDevice) {
}

CudaPoolingDeliverKernel::~CudaPoolingDeliverKernel() {
   cudnnDestroyPoolingDescriptor((cudnnPoolingDescriptor_t) params.poolingDescriptor);
   params.poolingDescriptor = nullptr;
   cudnnDestroyTensorDescriptor((cudnnTensorDescriptor_t) params.dataStoreDescriptor);
   params.dataStoreDescriptor = nullptr;
   cudnnDestroyTensorDescriptor((cudnnTensorDescriptor_t) params.gSynDescriptor);
   params.gSynDescriptor = nullptr;
}

void CudaPoolingDeliverKernel::setArgs(
      PVLayerLoc const * preLoc,
      PVLayerLoc const * postLoc,
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
      CudaBuffer * dataStoreBuffer,
      CudaBuffer * gSynBuffer,
      int channel) {

   pvErrorIf (preLoc->nx < postLoc->nx,
         "Pooling is not defined for one-to-many connections (pre->nx=%d, post->nx=%d\n",
         preLoc->nx, postLoc->nx);
   pvErrorIf (preLoc->ny < postLoc->ny,
         "Pooling is not defined for one-to-many connections (pre->ny=%d, post->ny=%d\n",
         preLoc->ny, postLoc->ny);

   cudnnStatus_t status;

   params.preLoc = preLoc;
   params.postLoc = postLoc;
   params.poolingMode = poolingMode;
   params.multiplier = (float) multiplier;

   int strideX = calcStride(preLoc->nx, postLoc->nx);
   int strideY = calcStride(preLoc->ny, postLoc->ny);

   cudnnPoolingDescriptor_t poolingDescriptor;
   status = cudnnCreatePoolingDescriptor(&poolingDescriptor);
   cudnnHandleError(status, "Create pooling descriptor");
   status = cudnnSetPooling2dDescriptor(
         poolingDescriptor,
         poolingMode,
         nypPost,
         nxpPost,
         0/*horizontal padding*/,
         0/*vertical padding*/,
         strideY,
         strideX
   );
   params.poolingDescriptor = (void*) poolingDescriptor;

   const PVHalo* preHalo = &preLoc->halo;
   params.diffX = calcBorderExcess(preLoc->nx, postLoc->nx, preHalo->lt, nxpPost);
   params.diffY = calcBorderExcess(preLoc->ny, postLoc->ny, preHalo->up, nypPost);
   cudnnTensorDescriptor_t dataStoreDescriptor;
   status = cudnnCreateTensorDescriptor(&dataStoreDescriptor);
   cudnnHandleError(status, "Create input tensor descriptor");
   status = cudnnSetTensor4dDescriptor(
         dataStoreDescriptor,
         CUDNN_TENSOR_NCHW, // PetaVision arrays are ordered NHWC; they will be permuted to NCHW inside do_run()
         CUDNN_DATA_FLOAT,
         preLoc->nbatch, //Number of images
         preLoc->nf, //Number of feature maps per image
         preLoc->ny + preHalo->up + preHalo->dn - 2*params.diffY, //Height of each feature map
         preLoc->nx + preHalo->lt + preHalo->rt - 2*params.diffX
         ); //Width of each feature map
   params.dataStoreDescriptor = (void*) dataStoreDescriptor;

   params.dataStore = (float*) dataStoreBuffer->getPointer();

   cudnnTensorDescriptor_t gSynDescriptor;
   status = cudnnCreateTensorDescriptor(&gSynDescriptor);
   cudnnHandleError(status, "Create input tensor descriptor");
   status = cudnnSetTensor4dDescriptor(
         gSynDescriptor,
         CUDNN_TENSOR_NCHW, // PetaVision arrays are ordered NHWC; they will be permuted to NCHW inside do_run()
         CUDNN_DATA_FLOAT,
         preLoc->nbatch, //Number of images
         postLoc->nf, //Number of feature maps per image
         postLoc->ny, //ny restricted
         postLoc->nx
   ); //nx restricted
   cudnnHandleError(status, "Set output tensor descriptor");
   params.gSynDescriptor = (void*) gSynDescriptor;

   int numNeuronsAcrossBatch = postLoc->nf*postLoc->ny*postLoc->nx*postLoc->nbatch;
   float * gSynHead = (float*) gSynBuffer->getPointer();
   params.gSyn = &gSynHead[channel*numNeuronsAcrossBatch];

   cudnnDataStore = device->createBuffer(dataStoreBuffer->getSize());

   size_t gSynSize = gSynBuffer->getSize();
   int numGSynNeurons = postLoc->nf*postLoc->ny*postLoc->nf*postLoc->nbatch;
   cudnnGSyn = device->createBuffer(numGSynNeurons);
}

int CudaPoolingDeliverKernel::calcBorderExcess(int preRestricted, int postRestricted, int border, int patchSizePostPerspective) {
   int preToPostScale = preRestricted/postRestricted;
   int borderNeeded = (patchSizePostPerspective-preToPostScale)/2;
   return border - borderNeeded;
}

int CudaPoolingDeliverKernel::calcManyScale(int preRestricted, int postRestricted) {
   return 1;
}

int CudaPoolingDeliverKernel::calcStride(int preRestricted, int postRestricted) {
   return preRestricted/postRestricted;
}

int CudaPoolingDeliverKernel::do_run() {
   float scalingFactor = 1.0f;

   int const blockSize = device->get_max_threads();

   // Permute PV-organized DataStore to CUDNN organization.
   PVLayerLoc const * preLoc = params.preLoc;
   PVHalo const * halo = &preLoc->halo;
   int const nxPreExt =preLoc->nx + halo->lt + halo->rt;
   int const nyPreExt = preLoc->ny + halo->dn + halo->up;
   int const nf = preLoc->nf;
   int const nbatch = params.preLoc->nbatch;
   //Calculate grid and work size
   int numNeurons = nbatch * nyPreExt * nxPreExt * nf;
   //Ceil to get all neurons
   int const gridSizePre = std::ceil((float)numNeurons/blockSize);
   float * cudnnDataStorePointer = (float*) cudnnDataStore->getPointer();
   callPermuteDatastorePVToCudnnKernel(gridSizePre, blockSize, params.dataStore, cudnnDataStorePointer, nbatch, nyPreExt, nxPreExt, nf, params.diffX, params.diffY);
   handleCallError("Permute DataStore PV to CUDNN");

   // Permute the PV-ordered GSyn channel to CUDNN ordering.
   int const nxPost = params.postLoc->nx;
   int const nyPost = params.postLoc->ny;
   int const nfPost = params.postLoc->nf;
   pvAssert(params.postLoc->nbatch == params.preLoc->nbatch);
   //Calculate grid and work size
   numNeurons = nbatch * nxPost * nyPost * nf;
   float* gSynPatchHead = params.gSyn;
   float * cudnnGSynPointer = (float*) cudnnGSyn->getPointer();
   //Ceil to get all neurons
   int const gridSizePost = std::ceil((float)numNeurons/(float)blockSize);
   callPermuteGSynPVToCudnnKernel(gridSizePost, blockSize, gSynPatchHead, cudnnGSynPointer, nbatch, nyPost, nxPost, nf, 1, 1);
   handleCallError("Permute GSyn PV to CUDNN");

   cudnnPoolingMode_t checkMode;
   int h,w,vPad,hPad,vStride,hStride;
   cudnnGetPooling2dDescriptor((cudnnPoolingDescriptor_t) params.poolingDescriptor,
   &checkMode, &h, &w, &vPad, &hPad, &vStride, &hStride);

   // Do the pooling
   cudnnStatus_t status = cudnnPoolingForward(
         (cudnnHandle_t) device->getCudnnHandle(),
         (cudnnPoolingDescriptor_t) params.poolingDescriptor,
         &params.multiplier,
         (cudnnTensorDescriptor_t) params.dataStoreDescriptor,
         cudnnDataStorePointer,
         &scalingFactor,
         (cudnnTensorDescriptor_t) params.gSynDescriptor,
         cudnnGSynPointer
   );
   cudnnHandleError(status, "Forward pooling run");

   // Permute the CUDNN-ordering GSyn back to PV ordering
   callPermuteGSynCudnnToPVKernel(gridSizePost, blockSize, gSynPatchHead, cudnnGSynPointer, nbatch, nyPost, nxPost, nf, 1, 1);
   handleCallError("Permute GSyn CUDNN back to PV");
   return 0;
}

} /* namespace PVCuda */
