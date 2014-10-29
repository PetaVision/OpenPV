#include "CudaRecvPre.hpp"
#include "../arch/cuda/cuda_util.hpp"
#include "conversions.hcu"

namespace PVCuda{

//Kernel code
__global__
void HyPerLayer_recv_pre(
   recv_pre_params params
){
   ////Shared memory buffers are declared
   //extern __shared__ char sharedMem[];
   //__shared__ float* postBuffer;
   //__shared__ float* preBuffer;
   //__shared__ long* localStartSourceExt;
   //postBuffer = (float*)sharedMem;
   //preBuffer = (float*)(&postBuffer[params.postBufNum]);
   //__shared__ long localStartSourceExt;
   //localStartSourceExt = (long*)(&preBuffer[params.preBufNum]);

   //Ordered this way because threads vary fastest in x, then y, then z
   //Mapped to petavision order of f, x, and y

   
   //int localX = blockDim.x;
   //int localY = blockDim.y;
   
   //int localXIndex = threadIdx.x;
   //int localYIndex = threadIdx.y;

   int groupXPost = (blockIdx.x * blockDim.x) + threadIdx.x;
   int groupYPost = (blockIdx.y * blockDim.y) + threadIdx.y;

   int postX = groupXPost * params.groupXSize;
   int postY = groupYPost * params.groupYSize;
   int kPostRes = kIndex(postX, postY, 0, params.postNxRes, params.postNyRes, params.postNf); 

   ////Clear post buffer
   //int postBufXSize = params.groupXSize * localX;
   //int postBufYStride = postBufXSize * params.postNf;
   //int postBufNumXF = params.groupXSize * params.postNf;
   //float* localGSynStart = &(postBuffer[localYIndex * params.groupYSize * postBufYStride + localXIndex * params.groupXSize * params.postNf]);

   //for(int ky = 0; ky < params.groupYSize; ky++){
   //   float* gSynStartY = localGSynStart + ky * postBufYStride;
   //   for(int kxf = 0; kxf < postBufNumXF; kxf++){
   //      gSynStartY[kxf] = 0;
   //   }
   //}
   //__syncthreads();

   //Calculate this local group's startPreExt
   int startPreExt = params.postToPreActivity[kPostRes];
   float* gSynStart = &(params.postGSyn[kPostRes]);

   //Loop through pre activity
   for(int kPreLocal = 0; kPreLocal < params.localPreSizeY * params.localPreSizeX * params.preNf; kPreLocal++){
      //Find x, y, and f index from kPreLocal
      int kPreXLocal = kxPos(kPreLocal, params.localPreSizeX, params.localPreSizeY, params.preNf);
      int kPreYLocal = kyPos(kPreLocal, params.localPreSizeX, params.localPreSizeY, params.preNf);
      int kPreFLocal = featureIndex(kPreLocal, params.localPreSizeX, params.localPreSizeY, params.preNf);
      int kPre = startPreExt + kPreYLocal * (params.preNxExt * params.preNf) + kPreXLocal * params.preNf + kPreFLocal;
      float a = params.preData[kPre] * params.dt_factor;
      if(a == 0) continue;


      ////GSynPatchStart is in local post space, need to translate to global post space
      size_t localGSynOffset = params.gSynPatchStart[kPreLocal];
      //size_t gSynOffset = params.gSynPatchStart[kPreLocal];

      int localXGSynOffset = kxPos(localGSynOffset, params.groupXSize, params.groupYSize, params.postNf);
      int localYGSynOffset = kyPos(localGSynOffset, params.groupXSize, params.groupYSize, params.postNf);
      int localFGSynOffset = featureIndex(localGSynOffset, params.groupXSize, params.groupYSize, params.postNf);
      //size_t gSynOffset = localYGSynOffset * postBufYStride + localXGSynOffset * params.postNf + localFGSynOffset;
      size_t gSynOffset = localYGSynOffset * params.sy + localXGSynOffset * params.postNf + localFGSynOffset;

      //Grab weight patches
      PVPatch patch = params.patches[kPreLocal];
      int nk = params.nfp * patch.nx;
      int ny = patch.ny;

      int kernelIndex;
      if(params.sharedWeights == 1){
         kernelIndex = params.patch2datalookuptable[kPre];
      }
      else{
         kernelIndex = kPre;
      }

      int wIdx = kernelIndex * params.nxp * params.nyp * params.nfp + patch.offset;

      for(int ky = 0; ky < ny; ky++){
         //float * gSynY = postBuffer + gSynOffset + ky * postBufYStride;
         float * gSynY = gSynStart + gSynOffset + ky * params.sy;
         float * weightY = &(params.weights[wIdx + ky*params.syw]);
         //Summing into post buffer indexed by localIndex
         for(int k = 0; k < nk; k++){
            gSynY[k] += a * weightY[k];
         }
      }
   }
   //__syncthreads();
   ////Copy post buf to gsyn start
   //float* gSynStart = &(params.postGSyn[kPostRes]);
   //for(int ky = 0; ky < params.groupYSize; ky++){
   //   float* globalGSynY = gSynStart + ky * params.sy;
   //   float* localGSynY = localGSynStart + ky * postBufYStride;
   //   for(int kxf = 0; kxf < postBufNumXF; kxf++){
   //      globalGSynY[kxf] = localGSynY[kxf];
   //   }
   //}
}


CudaRecvPre::CudaRecvPre(CudaDevice* inDevice):CudaKernel(inDevice){
}

CudaRecvPre::~CudaRecvPre(){
}

void CudaRecvPre::setArgs(
      int preNxExt,
      int preNyExt,
      int preNf,
      int postNxRes,
      int postNyRes,
      int postNf,

      int nxp,
      int nyp,
      int nfp,
      int groupXSize,
      int groupYSize,
      int localPreSizeX,
      int localPreSizeY,
      int localBufSizeX,
      int localBufSizeY,

      int sy,
      int syw,
      float dt_factor,
      int sharedWeights,

      /* PVPatch* */ CudaBuffer* patches,
      /* size_t* */  CudaBuffer* gSynPatchStart,

      /* long* */    CudaBuffer* postToPreActivity,
      /* float* */   CudaBuffer* preData,
      /* float* */   CudaBuffer* weights,
      /* float* */   CudaBuffer* postGSyn,
      /* int* */     CudaBuffer* patch2datalookuptable
   ){
   params.preNxExt = preNxExt;
   params.preNyExt = preNyExt;
   params.preNf = preNf;
   params.postNxRes = postNxRes;
   params.postNyRes = postNyRes;
   params.postNf = postNf;

   params.nxp = nxp;
   params.nyp = nyp;
   params.nfp = nfp;

   params.groupXSize = groupXSize;
   params.groupYSize = groupYSize;
   params.localPreSizeX = localPreSizeX;
   params.localPreSizeY = localPreSizeY;
   params.localBufSizeX = localBufSizeX;
   params.localBufSizeY = localBufSizeY;

   params.sy = sy;
   params.syw = syw;
   params.dt_factor = dt_factor;
   params.sharedWeights = sharedWeights;

   params.patches = (PVPatch*)patches->getPointer();
   params.gSynPatchStart = (size_t*)gSynPatchStart->getPointer();

   params.postToPreActivity = (long*)postToPreActivity->getPointer();
   params.preData = (float*)preData->getPointer();
   params.weights = (float*)weights->getPointer();
   params.postGSyn = (float*)postGSyn->getPointer();
   params.patch2datalookuptable = (int*)patch2datalookuptable->getPointer();

   setArgsFlag();
}

int CudaRecvPre::do_run(){

   params.preBufNum = params.localBufSizeY * params.localBufSizeX * params.nfp;

   printf("groupSize: (%d, %d)  blockSize: (%d, %d)  postNf: %d\n", params.groupXSize, params.groupYSize, block_size.x, block_size.y, params.postNf);
   params.postBufNum = (params.groupXSize*block_size.x) * (params.groupYSize*block_size.y) * params.postNf;
   //size_t sharedSize = sizeof(float) * (params.preBufNum + params.postBufNum);
   //size_t sharedSize = sizeof(float) * params.postBufNum;
   size_t sharedSize = 0;
   //printf("preBufNum: %d  postBufNum: %d\n", params.preBufNum, params.postBufNum);

   if(sharedSize > device->get_local_mem()){
      printf("run: given shared memory size of %zu is bigger than allowed shared memory size of %zu\n", sharedSize, device->get_local_mem());
      exit(-1);
   }
   
   //texReference.filterMode = cudaFilterModePoint;

   //cudaBindTexture(0, texReference, weights->getPointer(), weights->getSize());

   HyPerLayer_recv_pre<<<grid_size, block_size, sharedSize>>>(
      params
   );
   handleCallError();

   //cudaUnbindTexture(texReference);
   return 0;
}

}
