#include "HyPerLCAInternalStateBuffer.hpp"
#include "cudakernels/conversions.hcu"

namespace PVCuda {

__global__ void updateKernelHyPerLCAInternalStateBuffer(
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
      bool const selfInteract,
      double const *dtAdapt,
      float const tau,
      float const *accumulatedGSyn,
      float const *A) {

   int n = (blockIdx.x * blockDim.x) + threadIdx.x;
   if (n < numNeurons * nbatch) {
      int b = n / numNeurons; // batch index
      int k = n % numNeurons; // neuron index within batch element

      float exp_tau                     = (float)std::exp(-dtAdapt[b] / (double)tau);
      float *VBatch                     = V + b * numNeurons;
      float const *accumulatedGSynBatch = accumulatedGSyn + b * numNeurons;
      float const gSyn                  = accumulatedGSynBatch[k];
      // Activity is an extended buffer.
      float const *ABatch = A + b * (nx + rt + lt) * (ny + up + dn) * nf;

      int kex   = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      VBatch[k] = exp_tau * VBatch[k] + (1.0f - exp_tau) * (gSyn + selfInteract * ABatch[kex]);
   }
}

} // end namespace PVCuda

namespace PV {

void HyPerLCAInternalStateBuffer::runKernel() {
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
   PVCuda::updateKernelHyPerLCAInternalStateBuffer<<<currGridSize, currBlockSize, 0, cudaStream>>>(
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
         mSelfInteract,
         (double const *)mCudaDtAdapt->getPointer(),
         mScaledTimeConstantTau,
         (float const *)mAccumulatedGSyn->getCudaBuffer()->getPointer(),
         (float const *)mActivity->getCudaBuffer()->getPointer());
}

} // end namespace PV
