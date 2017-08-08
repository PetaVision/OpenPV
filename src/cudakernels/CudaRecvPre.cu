#include "CudaRecvPre.hpp"
#include "conversions.hcu"

namespace PVCuda {

// Kernel code
__global__ void HyPerLayer_recv_pre(recv_pre_params params, int batchIdx) {
   unsigned int kPreExt;
   float a;
   Patch patch;
   int wIdx;
   int numberShrunkenWeights;

   long tIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

   // Put this on cpu
   int fullPatchSize = params.nfp * params.nxp * params.nyp;

   if (params.isSparse) {
      if (tIndex >= fullPatchSize * params.numActive[batchIdx]) {
         return;
      }
   }
   else {
      if (tIndex >= fullPatchSize * params.numPreExt) {
         return;
      }
   }

   unsigned int neuronIndex = tIndex / fullPatchSize;

   int preBatchOffset = batchIdx * params.numPreExt;
   if (params.isSparse) {
      kPreExt = params.activeIndices[neuronIndex + preBatchOffset].index;
      a       = params.activeIndices[neuronIndex + preBatchOffset].value;
   }
   else {
      kPreExt = neuronIndex;
      a       = params.preData[kPreExt + preBatchOffset] * params.dt_factor;
   }

   if (a == 0) {
      return;
   }

   int kernelIndex;
   if (params.sharedWeights == 1) {
      kernelIndex = params.patch2datalookuptable[kPreExt];
   }
   else {
      kernelIndex = kPreExt;
   }

   // Grab weight patches
   patch                 = params.patches[kPreExt];
   wIdx                  = kernelIndex * fullPatchSize + patch.offset;
   numberShrunkenWeights = params.nfp * patch.nx * patch.ny;

   //__syncthreads();
   // patch may be shrunken, if thread oob, return
   int patchIndex = tIndex % fullPatchSize;
   if (patchIndex >= numberShrunkenWeights) {
      return;
   }

   int postBatchOffset = batchIdx * params.numPostRes;
   float *gSynStart    = params.postGSyn + postBatchOffset + params.gSynPatchStart[kPreExt];

   // Calculate what y row patchIndex is in
   int ky = kyPos(patchIndex, patch.nx, patch.ny, params.nfp);
   int kx = kxPos(patchIndex, patch.nx, patch.ny, params.nfp);
   int kf = featureIndex(patchIndex, patch.nx, patch.ny, params.nfp);
   int k  = kx * params.nfp + kf;

   float *gSynPtr  = gSynStart + ky * params.sy + k;
   float weightVal = params.weights[wIdx + ky * params.syw + k];

   // Multiply values
   float outVal = a * weightVal * params.dt_factor;

   // Atomic add into postGSyn
   atomicAdd(gSynPtr, outVal);
}

int CudaRecvPre::do_run() {

   size_t sharedSize = 0;

   checkSharedMemSize(sharedSize);

   for (int b = 0; b < params.nbatch; b++) {
      HyPerLayer_recv_pre<<<grid_size, block_size, sharedSize>>>(params, b);
   }

   return 0;
}

} // end namespace PVCuda
