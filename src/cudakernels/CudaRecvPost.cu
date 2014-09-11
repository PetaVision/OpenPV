#include "CudaRecvPost.hpp"
#include "../arch/cuda/cuda_util.hpp"
#include "../arch/cuda/device_util.hpp"

namespace PVCuda{

//Kernel code
__global__
void HyPerLayer_recv_post(recv_post_params params){
   ////Shared memory buffers are declared
   extern __shared__ char sharedMem[];
   //__shared__ float* preBuffer;
   __shared__ float* postBuffer;
   __shared__ float* weightsBuffer;
   //__shared__ long* localStartSourceExt;
   //preBuffer = (float*)sharedMem;
   //postBuffer = (float*)(&(preBuffer[params.preBufNum]));
   postBuffer = (float*)sharedMem;
   weightsBuffer = (float*)(&(postBuffer[params.postBufNum]));

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
   //__shared__ long localStartSourceExt;
   //if(localXIndex == 0 && localYIndex == 0 && localFIndex == 0){
   //   localStartSourceExt = params.startSourceExtBuf[kTargetRes];
   //}

   long startSourceExt = params.startSourceExtBuf[kTargetRes];

   int localIndex = kIndex(localXIndex, localYIndex, localFIndex, localX, localY, localF);

   postBuffer[localIndex] = 0;
      
   int numXfBuffer = params.localBufSizeX * params.nfp;
   int numWeightsBuffer = params.nxp * params.nfp;

   int xOffset = localXIndex * params.preToPostScaleX;
   //int yOffset = localYIndex * params.preToPostScaleY;

   //Wait for shared memory loads
   __syncthreads();

   for(int ky = 0; ky < params.nyp; ky++){
      //Copy global to local, do this with all threads
      //Pre buffer
      //if(localIndex < warpSize){
      //   for(int i = localIndex; i < numXfBuffer; i+= warpSize){
      //      preBuffer[i] = params.preData[localStartSourceExt + ky * params.sy + i];
      //   }
      //}

      //Weights
      if(localIndex < warpSize){
         for(int i = localIndex; i < numWeightsBuffer; i+= warpSize){
            weightsBuffer[i] = params.weights[wIdx + ky * params.syp + i];
         }
      }
      //The actual pre buffer index
      __syncthreads();

      float* activityY = &(params.preData[startSourceExt + ky * params.sy]);
      //float* activityY = &(preBuffer[xOffset * params.nfp]);
      //float* activityY = &(preBuffer[(ky+yOffset) * params.localBufSizeX * params.nfp + xOffset*params.nfp]);

      float* weightY = weightsBuffer;
      //float* weightY = &(params.weights[wIdx + ky * params.syp]);
      //pvpatch_accumulate_from_post(numPerStride, postAddr, activityY, weightY, dt_factor, (void*)0);

      //Summing into post buffer indexed by localIndex
      int k;
      for (k = 0; k < params.numPerStride; k++) {
         postBuffer[localIndex] += activityY[k]*weightY[k]*params.dt_factor;
         //postBuffer[localIndex] += activityY[k]*weightsBuffer[k]*params.dt_factor;
      }
      __syncthreads();
   }

   ////Sum into global memory
   params.postGsyn[kTargetRes] += postBuffer[localIndex];
}


CudaRecvPost::CudaRecvPost(CudaDevice* inDevice):CudaKernel(inDevice){
}

CudaRecvPost::~CudaRecvPost(){
}

void CudaRecvPost::setArgs(
      const int nxRes, //num post neurons
      const int nyRes,
      const int nf,

      const int nblt, //Border of orig
      const int nbrt, //Border of orig
      const int nbdn, //Border of orig
      const int nbup, //Border of orig

      const int nxp,
      const int nyp,
      const int nfp,

      const int localBufSizeX,
      const int localBufSizeY,
      const float preToPostScaleX,
      const float preToPostScaleY,

      const int sy,
      const int syp,
      const int numPerStride,
      const float dt_factor,
      const int sharedWeights,

      /* long* */  CudaBuffer* startSourceExtBuf,
      /* float* */ CudaBuffer* preData,
      /* float* */ CudaBuffer* weights,
      /* float* */ CudaBuffer* postGsyn,
      /* int* */   CudaBuffer* patch2datalookuptable
   ){
   params.nxRes = nxRes;
   params.nyRes = nyRes;
   params.nf = nf;

   params.nblt = nblt;
   params.nbrt = nbrt;
   params.nbdn = nbdn;
   params.nbup = nbup;

   params.nxp = nxp;
   params.nyp = nyp;
   params.nfp = nfp;

   params.localBufSizeX = localBufSizeX;
   params.localBufSizeY = localBufSizeY;
   params.preToPostScaleX = preToPostScaleX;
   params.preToPostScaleY = preToPostScaleY;

   params.sy = sy;
   params.syp = syp;
   params.numPerStride = numPerStride;
   params.dt_factor = dt_factor;
   params.sharedWeights = sharedWeights;

   params.startSourceExtBuf = (long*)startSourceExtBuf->getPointer();
   params.preData = (float*)preData->getPointer();
   params.weights = (float*)weights->getPointer();
   params.postGsyn = (float*)postGsyn->getPointer();
   params.patch2datalookuptable = (int*)patch2datalookuptable->getPointer();

   params.warpSize = device->get_warp_size();

   setArgsFlag();
}

int CudaRecvPost::run(){
   
   params.postBufNum = block_size.x * block_size.y * block_size.z;

   //int singlePreBufNum = params.localBufSizeX * params.nfp;
   //int singleWeightsBufNum = params.nxp * params.nfp;
   //params.numXfBufs = floor((device->get_local_mem()-sizeof(float)*params.postBufNum)/((singlePreBufNum + singleWeightsBufNum) * sizeof(float)));

   //params.numXfBufs = params.numXfBufs < params.nyp ? params.numXfBufs : params.nyp;  
   //params.numXfBufs = 1;

   //params.preBufNum = params.localBufSizeX * params.nfp;
   params.weightsBufNum = params.nxp * params.nfp;

   //size_t sharedSize = sizeof(float) * (params.preBufNum + params.postBufNum + params.weightsBufNum);
   size_t sharedSize = sizeof(float) * (params.postBufNum + params.weightsBufNum);

   if(sharedSize > device->get_local_mem()){
      printf("gpu post run: given shared memory size of %zu is bigger than allowed shared memory size of %zu\n", sharedSize, device->get_local_mem());
      exit(-1);
   }

   ////If sharedSize is greater than device's local memory, then numXFBufs should be greater than 1
   //assert(params.numXfBufs >= 1);

   if(block_size.x != 1){
      printf("gpu post run: numFLocal must be 1\n");
      exit(-1);
   }
   //if(block_size.z != 1){
   //   printf("gpu post run: numYLocal must be 1\n");
   //   exit(-1);
   //}

   //printf("Using %d buffers\n", params.numXfBufs);
   
   HyPerLayer_recv_post<<<grid_size, block_size, sharedSize>>>(params);
   handleCallError();

   return 0;
}

}
