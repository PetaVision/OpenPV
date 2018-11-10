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
      int const numChannels,
      float *V,
      bool const selfInteract,
      double *dtAdapt,
      float const tau,
      float const *GSynHead,
      float const *A) {

   int kIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
   if (kIndex < numNeurons * nbatch) {
      float const *GSynExc = GSynHead + 0 * numNeurons * nbatch;
      if (numChannels == 1) {
         int b = kIndex / numNeurons;
         int k = kIndex % numNeurons;

         float exp_tau             = (float)std::exp(-dtAdapt[b] / (double)tau);
         float *VBatch             = V + b * numNeurons;
         float const *GSynExcBatch = GSynExc + b * numNeurons;
         // Activity is an extended buffer.
         float const *ABatch = A + b * (nx + rt + lt) * (ny + up + dn) * nf;

         float const gSyn = GSynExcBatch[k]; // only one channel
         int kex          = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
         VBatch[k] = exp_tau * VBatch[k] + (1.0f - exp_tau) * (gSyn + selfInteract * ABatch[kex]);
      }
      else {
         float const *GSynInh = GSynHead + 1 * numNeurons * nbatch;
         int b                = kIndex / numNeurons;
         int k                = kIndex % numNeurons;

         float exp_tau             = (float)std::exp(-dtAdapt[b] / (double)tau);
         float *VBatch             = V + b * numNeurons;
         float const *GSynExcBatch = GSynExc + b * numNeurons;
         float const *GSynInhBatch = GSynInh + b * numNeurons;
         // Activity is an extended buffer.
         float const *ABatch = A + b * (nx + rt + lt) * (ny + up + dn) * nf;

         float const gSyn = GSynExcBatch[k] - GSynInhBatch[k];
         int kex          = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
         VBatch[k] = exp_tau * VBatch[k] + (1.0f - exp_tau) * (gSyn + selfInteract * ABatch[kex]);
      }
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
         params.numChannels,
         params.V,
         params.selfInteract,
         params.dtAdapt,
         params.tau,
         params.GSynHead,
         params.activity);
   return 0;
}

} // end namespace PVCuda
