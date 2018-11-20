#include "ISTAInternalStateBuffer.hpp"
#include "cudakernels/conversions.hcu"

namespace PVCuda {

__global__ void updateKernelISTAInternalState(
      int const nbatch,
      int const numNeurons,
      int const nx,
      int const ny,
      int const nf,
      int const lt,
      int const rt,
      int const dn,
      int const up,
      float *V,
      float const VThresh,
      double const *dtAdapt,
      float const tau,
      float const *accumulatedGSyn,
      float const *A) {

   int kIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
   if (kIndex < numNeurons * nbatch) {
      int b                             = kIndex / numNeurons;
      int k                             = kIndex % numNeurons;
      float *VBatch                     = V + b * numNeurons;
      float const *accumulatedGSynBatch = accumulatedGSyn + b * numNeurons;
      // Activity is an extended buffer.
      float const *ABatch = A + b * (nx + rt + lt) * (ny + up + dn) * nf;

      float const gSyn = accumulatedGSynBatch[k];
      int kex          = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      float sign       = 0.0f;
      if (ABatch[kex] != 0.0f) {
         sign = ABatch[kex] / fabsf(ABatch[kex]);
      }
      VBatch[k] += ((float)dtAdapt[b] / tau) * (gSyn - (VThresh * sign));
   }
}

} // end namespace PVCuda

namespace PV {

void ISTAInternalStateBuffer::runKernel() {
   PVLayerLoc const *loc           = getLayerLoc();
   int const nx                    = loc->nx;
   int const ny                    = loc->ny;
   int const nf                    = loc->nf;
   int const lt                    = loc->halo.lt;
   int const rt                    = loc->halo.rt;
   int const dn                    = loc->halo.dn;
   int const up                    = loc->halo.up;
   int const numNeurons            = nx * ny * nf;
   int const nbatch                = loc->nbatch;
   int const numNeuronsAcrossBatch = numNeurons * nbatch;
   int currBlockSize               = mCudaDevice->get_max_threads();
   cudaStream_t cudaStream         = mCudaDevice->getStream();
   // Ceil to get all weights
   int currGridSize = (int)ceil(((float)numNeuronsAcrossBatch) / currBlockSize);
   // Call function
   PVCuda::updateKernelISTAInternalState<<<currGridSize, currBlockSize, 0, cudaStream>>>(
         nbatch,
         numNeurons,
         nx,
         ny,
         nf,
         lt,
         rt,
         dn,
         up,
         (float *)getCudaBuffer()->getPointer(),
         mActivity->getVThresh(),
         (double const *)mCudaDtAdapt->getPointer(),
         mScaledTimeConstantTau,
         (float const *)mAccumulatedGSyn->getCudaBuffer()->getPointer(),
         (float const *)mActivity->getCudaBuffer()->getPointer());
}

} // end namespace PVCuda
