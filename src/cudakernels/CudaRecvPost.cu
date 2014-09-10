#include "CudaRecvPost.hpp"
#include "../arch/cuda/cuda_util.hpp"
#include "../arch/cuda/device_util.hpp"

namespace PVCuda{

//Kernel code
__global__
void HyPerLayer_recv_post(recv_post_params params){
   ////Shared memory buffers are declared
   extern __shared__ char sharedMem[];
   __shared__ float* preBuffer;
   __shared__ float* postBuffer;
   //__shared__ long* localStartSourceExt;
   preBuffer = (float*)sharedMem;
   postBuffer = (float*)(&(preBuffer[params.preBufNum]));

   //postBuffer = (float*)sharedMem;
   //preBuffer = (float*)(&postBuffer[params.postBufNum]);
   //__shared__ long localStartSourceExt;
   //localStartSourceExt = (long*)(&preBuffer[params.preBufNum]);

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

   //Change restricted to extended post neuron
   int kTargetExt = kIndexExtended(kTargetRes, params.nxRes, params.nyRes, params.nf, params.nblt, params.nbrt, params.nbdn, params.nbup);

   int kernelIndex;
   if(params.sharedWeights == 1){
      kernelIndex = params.patch2datalookuptable[kTargetExt];
   }
   else{
      kernelIndex = kTargetExt;
   }
   int wIdx = kernelIndex * params.nxp * params.nyp * params.nfp;

   //Get top left most neuron in the group
   __shared__ long localStartSourceExt;
   if(localXIndex == 0 && localYIndex == 0 && localFIndex == 0){
      localStartSourceExt = params.startSourceExtBuf[kTargetRes];
   }

   long startSourceExt = params.startSourceExtBuf[kTargetRes];

   int localIndex = kIndex(localXIndex, localYIndex, localFIndex, localX, localY, localF);

   postBuffer[localIndex] = 0;
      
   int numXfBuffer = params.localBufSizeX * params.nfp;

   int xOffset = localXIndex * params.preToPostScaleX;
   //int yOffset = localYIndex * params.preToPostScaleY;

   //Wait for shared memory loads
   __syncthreads();


   for(int ky = 0; ky < params.nyp; ky++){

      //Copy global to local, do this with all threads
      //This function has thread sync in it
      //Find total number of threads working on this copy
      if(localIndex < warpSize){
         for(int i = localIndex; i < numXfBuffer; i+= warpSize){
            //if(ky == 5 && i == 47){
            //   printf("Break here\n");
            //}
            preBuffer[i] = params.preData[localStartSourceExt + ky * params.sy + i];
         }
      }
      __syncthreads();

      //float* activityY = &(params.preData[startSourceExt + ky * params.sy]);
      float* activityY = &(preBuffer[xOffset * params.nfp]);
      //float* activityY = &(preBuffer[(ky+yOffset) * params.localBufSizeX * params.nfp + xOffset*params.nfp]);

      float* weightY = &(params.weights[wIdx + ky * params.syp]);
      //pvpatch_accumulate_from_post(numPerStride, postAddr, activityY, weightY, dt_factor, (void*)0);

      //Summing into post buffer indexed by localIndex
      int k;
      for (k = 0; k < params.numPerStride; k++) {
         //if(activityY[k] != preBuffer[xOffset * params.nfp + k]){
         //   printf("Break here\n");
         //}
         //float weightVal = tex1Dfetch(texReference, (wIdx + ky*syp) + k);
         //float weightVal = weights[wIdx+ky*syp+k];
         //*gSynPatchPos += activityY[k]*weightY[k]*params.dt_factor;
         postBuffer[localIndex] += activityY[k]*weightY[k]*params.dt_factor;
      //   params.postGsyn[kTargetRes] += activityY[k]*weightY[k]*params.dt_factor;
         //*postAddr += activityY[k]*weightVal*dt_factor;
      }
      __syncthreads();
   }

   ////Barrier to make sure the work group's postbuffer is set
   __syncthreads();

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

   params.preBufNum = params.localBufSizeX * params.nfp;
   params.postBufNum = block_size.x * block_size.y * block_size.z;
   size_t sharedSize = sizeof(float) * (params.preBufNum + params.postBufNum);
   //size_t sharedSize = sizeof(float) * params.postBufNum;
   //printf("preBufNum: %d  postBufNum: %d\n", params.preBufNum, params.postBufNum);

   if(sharedSize > device->get_local_mem()){
      printf("run: given shared memory size of %zu is bigger than allowed shared memory size of %zu\n", sharedSize, device->get_local_mem());
      exit(-1);
   }
   
   //texReference.filterMode = cudaFilterModePoint;

   //cudaBindTexture(0, texReference, weights->getPointer(), weights->getSize());

   HyPerLayer_recv_post<<<grid_size, block_size, sharedSize>>>(
      params
   );
   handleCallError();

   //cudaUnbindTexture(texReference);
   return 0;
}

}
