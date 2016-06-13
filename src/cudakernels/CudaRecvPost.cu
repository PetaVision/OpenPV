#include "CudaRecvPost.hpp"
#include "../arch/cuda/cuda_util.hpp"
#include "conversions.hcu"

namespace PVCuda{

#ifdef PV_USE_CUDNN
#include <cudnn.h>

//Function to change PV representation to CUDNN reprsentation
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

#endif // PV_USE_CUDNN

//Kernel code
__global__
void HyPerLayer_recv_post(recv_post_params params, int batch){
   ////Shared memory buffers are declared
   extern __shared__ char sharedMem[];
   __shared__ float* preBuffer;
   __shared__ float* postBuffer;
   __shared__ float* weightsBuffer;

   postBuffer = (float*)sharedMem;
   weightsBuffer = (float*)(&(postBuffer[params.postBufNum]));

   if(params.preDataLocal){
      preBuffer = (float*)(&(weightsBuffer[params.weightsBufNum]));
   }

   //Ordered this way because threads vary fastest in x, then y, then z
   //Mapped to petavision order of f, x, and y

   int localF = blockDim.x;
   int localX = blockDim.y;
   int localY = blockDim.z;
   
   int localFIndex = threadIdx.x;
   int localXIndex = threadIdx.y;
   int localYIndex = threadIdx.z;

   int fTargetRes = (blockIdx.x * blockDim.x) + threadIdx.x;
   int xTargetRes = (blockIdx.y * blockDim.y) + threadIdx.y;
   int yTargetRes = (blockIdx.z * blockDim.z) + threadIdx.z;

   ////Calculate kTargetRes based on x, y, and f
   int kTargetRes = kIndex(xTargetRes, yTargetRes, fTargetRes, params.nxRes, params.nyRes, params.nf);

   int kTargetExt = kIndexExtended(kTargetRes, params.nxRes, params.nyRes, params.nf, params.nblt, params.nbrt, params.nbdn, params.nbup);

   //Each wIdx should be shared since each workgroup convolves one weight kernel
   __shared__ int wIdx;
   if(localXIndex == 0 && localYIndex == 0){
      //Change restricted to extended post neuron
      int kernelIndex;
      if(params.sharedWeights == 1){
         kernelIndex = params.patch2datalookuptable[kTargetExt];
      }
      else{
         kernelIndex = kTargetExt;
      }
      wIdx = kernelIndex * params.nxp * params.nyp * params.nfp;
   }

   //Get top left most neuron in the group
   __shared__ long localStartSourceExt;
   long startSourceExt;
   if(params.preDataLocal){
      if(localXIndex == 0 && localYIndex == 0 && localFIndex == 0){
         localStartSourceExt = params.startSourceExtBuf[kTargetRes];
      }
   }
   else{
      startSourceExt = params.startSourceExtBuf[kTargetRes];
   }

   int localIndex = kIndex(localXIndex, localYIndex, localFIndex, localX, localY, localF);

   postBuffer[localIndex] = 0;
      
   int numXfBuffer = params.localBufSizeX * params.nfp;
   int numWeightsBuffer = params.nxp * params.nfp;

   int xOffset = localXIndex * params.preToPostScaleX;
   //int yOffset = localYIndex * params.preToPostScaleY;

   int numCopyThreads = localF * localX * localY < warpSize ? localF * localX * localY : warpSize;
   
   //Wait for shared memory loads
   __syncthreads();

   int preBatchOffset = batch * (params.preNx + params.preNblt + params.preNbrt) * (params.preNy + params.preNbup + params.preNbdn) * params.preNf; 

   for(int ky = 0; ky < params.nyp; ky++){
      //Copy global to local, do this with all threads
      if(params.preDataLocal){
         //Pre buffer
         if(localIndex < numCopyThreads){
            for(int i = localIndex; i < numXfBuffer; i+= numCopyThreads){
               preBuffer[i] = params.preData[preBatchOffset + localStartSourceExt + ky * params.sy + i];
            }
         }
      }

      //Weights
      if(localIndex < numCopyThreads){
         for(int i = localIndex; i < numWeightsBuffer; i+= numCopyThreads){
            weightsBuffer[i] = params.weights[wIdx + ky * params.syp + i];
         }
      }
      //The actual pre buffer index
      __syncthreads();

      float* activityY;
      if(params.preDataLocal){
         activityY = &(preBuffer[xOffset * params.nfp]);
      }
      else{
         activityY = &(params.preData[preBatchOffset + startSourceExt + ky * params.sy]);
      }

      float* weightY = weightsBuffer;
      //float* weightY = &(params.weights[wIdx + ky * params.syp]);

      //Summing into post buffer indexed by localIndex
      int k;
      for (k = 0; k < params.numPerStride; k++) {
         postBuffer[localIndex] += activityY[k]*weightY[k]*params.dt_factor;
      }
      __syncthreads();
   }
   ////Sum into global memory
   int postBatchOffset = batch * params.nxRes * params.nyRes * params.nf; 
   params.postGsyn[postBatchOffset + kTargetRes] += postBuffer[localIndex];
}


#ifdef PV_USE_CUDNN
void CudaRecvPost::permuteDatastorePVToCudnn(){
   //Ext pre activity
   int ny = params.preNy + params.preNbup + params.preNbdn;
   int nx = params.preNx + params.preNblt + params.preNbrt;
   int nf = params.preNf;
   int nbatch = params.nbatch;

   //Calculate grid and work size
   int numNeurons = nbatch * ny * nx * nf;
   int blockSize = device->get_max_threads();
   //Ceil to get all weights
   int gridSize = ceil((float)numNeurons/blockSize);

   device->syncDevice();

   //Call function
   //Datastore will never get reshaped, so manyScale will always be 1
   CudaPermutePVToCudnn<<<gridSize, blockSize, 0, device->getStream()>>>(params.cudnn_preData, params.preData, nbatch, ny, nx, nf, 1, 1, params.diffX, params.diffY);
   handleCallError("Permute PV to CUDNN");
}

void CudaRecvPost::permuteGSynPVToCudnn(int channel){
   //Res post activity
   int ny = params.nyRes;
   int nx = params.nxRes;
   int nf = params.nf;
   int nbatch = params.nbatch;

   //Calculate grid and work size
   int numNeurons = nbatch * ny * nx * nf;
   float* gSynPatchHead = &(params.postGsyn[numNeurons * channel]);

   int blockSize = device->get_max_threads();
   //Ceil to get all weights
   int gridSize = ceil((float)numNeurons/blockSize);
   //Call function
   CudaPermutePVToCudnn<<<gridSize, blockSize, 0, device->getStream()>>>(params.cudnn_gSyn, gSynPatchHead, nbatch, ny, nx, nf, params.manyScaleX, params.manyScaleY, 0, 0);
   handleCallError("Permute GSyn PV to CUDNN");
}

void CudaRecvPost::permuteGSynCudnnToPV(int channel){
   //Res post activity
   int ny = params.nyRes;
   int nx = params.nxRes;
   int nf = params.nf;
   int nbatch = params.nbatch;

   //Calculate grid and work size
   int numNeurons = nbatch * ny * nx * nf;
   float* gSynPatchHead = &(params.postGsyn[numNeurons * channel]);

   int blockSize = device->get_max_threads();
   //Ceil to get all weights
   int gridSize = ceil((float)numNeurons/blockSize);
   //Call function
   CudaPermuteCudnnToPV<<<gridSize, blockSize, 0, device->getStream()>>>(gSynPatchHead, params.cudnn_gSyn, nbatch, ny, nx, nf, params.manyScaleX, params.manyScaleY);
   handleCallError("Permute GSyn CUDNN to PV");
}

//void CudaRecvPost::permuteWeightsPVToCudnn(){
//   //outFeatures is number of kernels
//   int outFeatures = params.nf;
//
//   //Rest is patch sizes
//   int ny = params.nyp;
//   int nx = params.nxp;
//   int inFeatures = params.nfp;
//
//   //Calculate grid and work size
//   int numWeights = outFeatures * params.manyScaleX * params.manyScaleY * ny * nx * inFeatures;
//   int blockSize = device->get_max_threads();
//   //Ceil to get all weights
//   int gridSize = ceil((float)numWeights/blockSize);
//   //Call function
//   CudaPermuteWeightsPVToCudnn<<<gridSize, blockSize, 0, device->getStream()>>>(params.cudnn_weights, params.weights, outFeatures, ny, nx, inFeatures, params.manyScaleX, params.manyScaleY);
//   handleCallError("Permute weights PV to CUDNN");
//}

#endif // PV_USE_CUDNN

}  // end namespace PVCuda
