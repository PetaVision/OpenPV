#include "CudaUpdateISTAInternalState.hpp"
#include "conversions.hcu"

namespace PVCuda {

__global__ void HyPerLCAInternalState_update_state(
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
      double *dtAdapt,
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

int CudaUpdateISTAInternalState::do_run() {
   int currBlockSize = device->get_max_threads();
   // Ceil to get all weights
   int currGridSize = ceil(((float)params.numNeurons * params.nbatch) / currBlockSize);
   // Call function
   HyPerLCAInternalState_update_state<<<currGridSize, currBlockSize, 0, device->getStream()>>>(
         params.nbatch,
         params.numNeurons,
         params.nx,
         params.ny,
         params.nf,
         params.lt,
         params.rt,
         params.dn,
         params.up,
         params.V,
         params.VThresh,
         params.dtAdapt,
         params.tau,
         params.accumulatedGSyn,
         params.activity);
   return 0;
}

} // end namespace PVCuda
