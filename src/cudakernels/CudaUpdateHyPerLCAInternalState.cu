#include "CudaUpdateHyPerLCAInternalState.hpp"
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
      bool const selfInteract,
      double *dtAdapt,
      float const tau,
      float const *accumulatedGSyn,
      float const *A) {

   int kIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
   if (kIndex < numNeurons * nbatch) {
      int b = kIndex / numNeurons;
      int k = kIndex % numNeurons;

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

int CudaUpdateHyPerLCAInternalState::do_run() {
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
         params.selfInteract,
         params.dtAdapt,
         params.tau,
         params.accumulatedGSyn,
         params.activity);
   return 0;
}

} // end namespace PVCuda
