#include "CudaRecvPost.hpp"
#include "../arch/cuda/cuda_util.hpp"
#include "../arch/cuda/device_util.hpp"

namespace PVCuda{

//Declaring texture reference
//texture<float, 1, cudaReadModeElementType> texReference;
//Note that this reference is not a member variable, so it must be bound and unbound at the beginning and end of run()


//__device__
//int kxPos(int k, int nx, int ny, int nf)
//{
//   return (k/nf) % nx;
//}
//
//__device__
//int kyPos(int k, int nx, int ny, int nf)
//{
//   return k / (nx*nf);
//}
//
//__device__
//int featureIndex(int k, int nx, int ny, int nf)
//{
//   return k % nf;
//}

//__device__
//int kIndex(int kx, int ky, int kf, int nx, int ny, int nf)
//{
//   return kf + (kx + ky * nx) * nf;
//}
//
//__device__
//int kIndexExtended(int k, int nx, int ny, int nf, int nb)
//{
//   const int kx_ex = nb + kxPos(k, nx, ny, nf);
//   const int ky_ex = nb + kyPos(k, nx, ny, nf);
//   const int kf = featureIndex(k, nx, ny, nf);
//   return kIndex(kx_ex, ky_ex, kf, nx + 2*nb, ny + 2*nb, nf);
//}
//
////A function that copys src to dest elements using all avaliable threads in a block
//__device__
//void work_group_copy(float* dest, float* src, int numElements, int warpSize, int threadIndex){
//   if(threadIndex < warpSize){
//      for(int i = threadIndex; i < numElements; i+= warpSize){
//         dest[i] = src[i];
//      }
//   }
//}

//__device__
//int pvpatch_accumulate_from_post(int nk, float * v, float * a, float * w, float dt_factor, void * auxPtr) {
//   int status = 0;
//   int k;
//   for (k = 0; k < nk; k++) {
//      *v += a[k]*w[k]*dt_factor;
//      //dv = dv + a[k]*w[k];
//   }
//   //*v = *v + dt_factor*dv;
//   return status;
//}

//Kernel code
__global__
void HyPerLayer_recv_post(
   recv_post_params params
){

      ////Shared memory buffers are declared
      extern __shared__ char sharedMem[];
      __shared__ float* postBuffer;
      //__shared__ float* preBuffer;
      //__shared__ long* localStartSourceExt;
      postBuffer = (float*)sharedMem;
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

      //int kTargetRes = blockIdx.x;

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
      //if(localXIndex == 0 && localYIndex == 0 && localFIndex == 0){
      //   localStartSourceExt = params.startSourceExtBuf[kTargetRes];
      //}

      long startSourceExt = params.startSourceExtBuf[kTargetRes];

      int localIndex = kIndex(localXIndex, localYIndex, localFIndex, localX, localY, localF);

      postBuffer[localIndex] = 0;
         
      //int numXfBuffer = params.localBufSizeX * params.nfp;

      //////Match preBuffer to indPreData, need to find x and y offsets
      //
      ////TODO find a better size of pre buffer to see if you can do multiple lines at a time
      //for(int ky = 0; ky < params.localBufSizeY ; ky++){
      //   float* preDataY = &(params.preData[*localStartSourceExt + ky * params.sy]);
      //   float* preBufferY = &(preBuffer[ky * params.localBufSizeX * params.nfp]);
      //   work_group_copy(preBufferY, preDataY, numXfBuffer, warpSize, localIndex); 
      //}

      //Wait for shared memory loads
      __syncthreads();

      //int xOffset = localXIndex * params.preToPostScaleX;
      //int yOffset = localYIndex * params.preToPostScaleY;


      for(int ky = 0; ky < params.nyp; ky++){


         //Copy global to local, do this with all threads
         //This function has thread sync in it
         //Find total number of threads working on this copy
         
         //int numThreads = localX * localY * localF;
         //int numThreadsPart = numThreads < warpSize ? numThreads : warpSize;
         //float* preDataY = &(params.preData[*localStartSourceExt + ky * params.sy]);
         //if(localIndex < numThreadsPart){
         //   for(int i = localIndex; i < numXfBuffer; i+= numThreadsPart){
         //      preBuffer[i] = preDataY[i];
         //   }
         //}
         //__syncthreads();

         //work_group_copy(preBuffer, &(params.preData[localStartSourceExt + ky * params.sy]), numXfBuffer, warpSize, localIndex);
         //__syncthreads();

         float* activityY = &(params.preData[startSourceExt + ky * params.sy]);
         //float* activityY = &(preBuffer[xOffset * params.nfp]);
         //float* activityY = &(preBuffer[(ky+yOffset) * params.localBufSizeX * params.nfp + xOffset*params.nfp]);

         float* weightY = &(params.weights[wIdx + ky * params.syp]);
         //pvpatch_accumulate_from_post(numPerStride, postAddr, activityY, weightY, dt_factor, (void*)0);

         //Summing into post buffer indexed by localIndex
         int k;
         for (k = 0; k < params.numPerStride; k++) {
            //if(dataActivityY[k] != activityY[k]){
            //   printf("Break here\n");
            //}
            //float weightVal = tex1Dfetch(texReference, (wIdx + ky*syp) + k);
            //float weightVal = weights[wIdx+ky*syp+k];
            //*gSynPatchPos += activityY[k]*weightY[k]*params.dt_factor;
            postBuffer[localIndex] += activityY[k]*weightY[k]*params.dt_factor;
         //   params.postGsyn[kTargetRes] += activityY[k]*weightY[k]*params.dt_factor;
            //*postAddr += activityY[k]*weightVal*dt_factor;
         }
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

   params.preBufNum = params.localBufSizeY * params.localBufSizeX * params.nfp;
   params.postBufNum = block_size.x * block_size.y * block_size.z;
   //size_t sharedSize = sizeof(float) * (params.preBufNum + params.postBufNum);
   size_t sharedSize = sizeof(float) * params.postBufNum;
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
