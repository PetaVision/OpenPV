#include "CudaRecvPre.hpp"
#include "../arch/cuda/cuda_util.hpp"
#include "conversions.hcu"

namespace PVCuda{

//Kernel code
__global__
void HyPerLayer_recv_pre(
   recv_pre_params params,
   int batchIdx
){
   unsigned int kPreExt;
   float a;
   PVPatch patch;
   int wIdx;
   int numberShrunkenWeights;

   long tIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

   //Put this on cpu
   int fullPatchSize = params.nfp * params.nxp * params.nyp;

   if(params.isSparse){
      if(tIndex >= fullPatchSize * params.numActive[batchIdx]){
         return;
      }
   }
   else{
      if(tIndex >= fullPatchSize * params.numPreExt){
         return;
      }
   }

   unsigned int neuronIndex = tIndex / fullPatchSize;

   int preBatchOffset = batchIdx * params.numPreExt;
   if(params.isSparse){
      kPreExt = params.activeIndices[neuronIndex + preBatchOffset];
   }
   else{
      kPreExt = neuronIndex;
   }
   a = params.preData[kPreExt + preBatchOffset] * params.dt_factor;
   int kernelIndex;
   if(params.sharedWeights == 1){
      kernelIndex = params.patch2datalookuptable[kPreExt];
   }
   else{
      kernelIndex = kPreExt;
   }
   //Grab weight patches
   patch = params.patches[kPreExt];
   wIdx = kernelIndex * fullPatchSize + patch.offset;
   numberShrunkenWeights = params.nfp * patch.nx * patch.ny;

   //__syncthreads();

   if(a == 0) return;
   //patch may be shrunken, if thread oob, return
   int patchIndex = tIndex % fullPatchSize;
   if(patchIndex >= numberShrunkenWeights){
      return;
   }

   int postBatchOffset = batchIdx * params.numPostRes;
   float* gSynStart = params.postGSyn + postBatchOffset + params.gSynPatchStart[kPreExt];

   //Calculate what y row patchIndex is in
   int ky = kyPos(patchIndex, patch.nx, patch.ny, params.nfp);
   int kx = kxPos(patchIndex, patch.nx, patch.ny, params.nfp); 
   int kf = featureIndex(patchIndex, patch.nx, patch.ny, params.nfp); 
   int k = kx * params.nfp + kf;

   float * gSynPtr = gSynStart + ky*params.sy + k;
   float weightVal = params.weights[wIdx + ky*params.syw + k];

   //Multiply values
   float outVal = a * weightVal;

   //Atomic add into postGSyn
   atomicAdd(gSynPtr, outVal);
}


CudaRecvPre::CudaRecvPre(CudaDevice* inDevice):CudaKernel(inDevice){
}

CudaRecvPre::~CudaRecvPre(){
}

void CudaRecvPre::setArgs(
      int nbatch,
      int numPreExt,
      int numPostRes,
      int nxp,
      int nyp,
      int nfp,

      int sy,
      int syw,
      float dt_factor,
      int sharedWeights,

      /* PVPatch* */ CudaBuffer* patches,
      /* size_t* */  CudaBuffer* gSynPatchStart,

      /* float* */   CudaBuffer* preData,
      /* float* */   CudaBuffer* weights,
      /* float* */   CudaBuffer* postGSyn,
      /* int* */     CudaBuffer* patch2datalookuptable,

      bool isSparse,
      /*unsigned long*/ CudaBuffer* numActive,
      /*unsigned int*/ CudaBuffer* activeIndices
   ){
   params.nbatch = nbatch;
   params.numPreExt = numPreExt;
   params.numPostRes = numPostRes;

   params.nxp = nxp;
   params.nyp = nyp;
   params.nfp = nfp;

   params.sy = sy;
   params.syw = syw;
   params.dt_factor = dt_factor;
   params.sharedWeights = sharedWeights;

   params.patches = (PVPatch*)patches->getPointer();
   params.gSynPatchStart = (size_t*)gSynPatchStart->getPointer();

   params.preData = (float*)preData->getPointer();
   params.weights = (float*)weights->getPointer();
   params.postGSyn = (float*)postGSyn->getPointer();
   params.patch2datalookuptable = (int*)patch2datalookuptable->getPointer();

   params.isSparse = isSparse;
   if(activeIndices){
      params.numActive = (long*)numActive->getPointer();
      params.activeIndices = (unsigned int*)activeIndices->getPointer();
   }
   else{
      params.activeIndices = NULL;
      params.numActive = NULL;
   }

   setArgsFlag();
}

int CudaRecvPre::do_run(){

   size_t sharedSize = 0;

   if(sharedSize > device->get_local_mem()){
      printf("run: given shared memory size of %zu is bigger than allowed shared memory size of %zu\n", sharedSize, device->get_local_mem());
      exit(-1);
   }

   for(int b = 0; b < params.nbatch; b++){
      HyPerLayer_recv_pre<<<grid_size, block_size, sharedSize>>>(
         params,
         b
      );
   }

   handleCallError("Cuda recv pre run");

   return 0;
}

}
