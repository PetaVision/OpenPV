/*
 * CudaKernel.cu
 *
 *  Created on: Aug 6, 2014
 *      Author: Sheng Lundquist
 */

#include "CudaKernel.hpp"
 
#ifdef PV_USE_CUDNN
#include <cudnn.h>

namespace PVCuda {

//Function to change PV representation to CUDNN representation
//Does 2 things: permutate ordering from [outFeature, ny, nx, inFeature] to [outFeature, inFeature, ny, nx]
//Reshapes the matrix if manyScale > 1 to map different "many" kernels into feature dimension
//Coalesced in input
__global__
void CudaPermutePVToCudnn(float* dest, float* src, int outFeatures, int ny, int nx, int inFeatures, int manyScaleX, int manyScaleY, int cropX, int cropY){
   //parameter dimensions are in source PV format
   int destNx = (nx-2*cropX)/manyScaleX;
   int destNy = (ny-2*cropY)/manyScaleY;
   int destInFeatures = inFeatures*manyScaleX*manyScaleY;

   int kSrc = (blockIdx.x * blockDim.x) + threadIdx.x;
   if(kSrc < outFeatures * ny * nx * inFeatures){
      int kOF = kSrc/(ny*nx*inFeatures);
      int kY  = (kSrc % (ny*nx*inFeatures))/(nx*inFeatures);
      int kX  = (kSrc % (nx*inFeatures))/inFeatures;
      int kIF = (kSrc % inFeatures);

      //check if in bounds
      if(kX < cropX || kX >= nx-cropX){ 
         return;
      }
      else{
         kX = kX - cropX;
      }
      if(kY < cropY || kY >= ny-cropY){
         return;
      }
      else{
         kY = kY - cropY;
      }

      //Recalculate x, y, and f based on manyScale
      kIF = kIF + inFeatures * (kX % manyScaleX + (kY % manyScaleY) * manyScaleX);
      kX = kX/manyScaleX;
      kY = kY/manyScaleY;

      int sOF = destInFeatures * destNy * destNx;
      int sIF = destNy * destNx;
      int sY  = destNx;

      int kDest = kOF * sOF + kIF * sIF + kY * sY + kX;

      dest[kDest] = src[kSrc];
   }
}

//Weights need to be reversed for cudnn
//No need to account for many because the PV representation matches with how gsyn was reshaped.
__global__
void CudaPermuteWeightsPVToCudnn(float* dest, float* src, int outFeatures, int ny, int nx, int inFeatures, int manyScaleX, int manyScaleY){
   //Parameter dimensions are PV source dimensions
   int kSrc = (blockIdx.x * blockDim.x) + threadIdx.x;
   if(kSrc < outFeatures * manyScaleX * manyScaleY * ny * nx * inFeatures){
      int kOF = kSrc/(ny*nx*inFeatures);
      int kY  = (kSrc % (ny*nx*inFeatures))/(nx*inFeatures);
      int kX  = (kSrc % (nx*inFeatures))/inFeatures;
      int kIF = (kSrc % inFeatures);

      int sOF = inFeatures * ny * nx;
      int sIF = ny * nx;
      int sY  = nx;

      int kDest = kOF * sOF + kIF * sIF + (ny-kY-1) * sY + (nx-kX-1);

      dest[kDest] = src[kSrc];
   }
}

__global__
void CudaPermuteCudnnToPV(float* dest, float* src, int outFeatures, int ny, int nx, int inFeatures, int manyScaleX, int manyScaleY){
   //parameter dimensions are in dest PV format
   int srcNx = nx/manyScaleX;
   int srcNy = ny/manyScaleY;
   int srcInFeatures = inFeatures*manyScaleX*manyScaleY;

   int kDest = (blockIdx.x * blockDim.x) + threadIdx.x;
   if(kDest < outFeatures * ny * nx * inFeatures){
      int kOF = kDest/(ny*nx*inFeatures);
      int kY  = (kDest % (ny*nx*inFeatures))/(nx*inFeatures);
      int kX  = (kDest % (nx*inFeatures))/inFeatures;
      int kIF = (kDest % inFeatures);

      //Recalculate x, y, and f based on manyScale
      kIF = kIF + inFeatures * (kX % manyScaleX + (kY % manyScaleY) * manyScaleX);
      kX = kX/manyScaleX;
      kY = kY/manyScaleY;

      int sOF = srcInFeatures * srcNy * srcNx;
      int sIF = srcNy * srcNx;
      int sY  = srcNx;

      int kSrc = kOF * sOF + kIF * sIF + kY * sY + kX;

      dest[kDest] = src[kSrc];
   }
}

void CudaKernel::callPermuteDatastorePVToCudnnKernel(int gridSize, int blockSize, float * pvBuffer, float * cudnnBuffer, int nbatch, int ny, int nx, int nf, int diffX, int diffY) {
   //Datastore will never get reshaped, so manyScale will always be 1
   CudaPermutePVToCudnn<<<gridSize, blockSize, 0, device->getStream()>>>(cudnnBuffer, pvBuffer, nbatch, ny, nx, nf, 1, 1, diffY, diffX);
}

void CudaKernel::callPermuteGSynPVToCudnnKernel(int gridSize, int blockSize, float * pvBuffer, float * cudnnBuffer, int nbatch, int ny, int nx, int nf, int manyScaleX, int manyScaleY) {
   CudaPermutePVToCudnn<<<gridSize, blockSize, 0, device->getStream()>>>(cudnnBuffer, pvBuffer, nbatch, ny, nx, nf, manyScaleX, manyScaleY, 0, 0);
}

void CudaKernel::callPermuteGSynCudnnToPVKernel(int gridSize, int blockSize, float * pvBuffer, float * cudnnBuffer, int nbatch, int ny, int nx, int nf, int manyScaleX, int manyScaleY) {
   CudaPermuteCudnnToPV<<<gridSize, blockSize, 0, device->getStream()>>>(pvBuffer, cudnnBuffer, nbatch, ny, nx, nf, manyScaleX, manyScaleY);
}

} // end namespace PVCuda

#endif // PV_USE_CUDNN
