#include "CudaUpdateMomentumLCAInternalState.hpp"
#include "conversions.hcu"

namespace PVCuda {

__global__ void MomentumLCAInternalState_update_state(
      const int nbatch,
      const int numNeurons,
      const int nx,
      const int ny,
      const int nf,
      const int lt,
      const int rt,
      const int dn,
      const int up,
      const int numChannels,
      float *V,
      float *prevDrive,
      const bool selfInteract,
      double *dtAdapt,
      const float tau,
      const float LCAMomentumRate,
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
         float const gSyn          = GSynExcBatch[k]; // only one channel
         float *prevDriveBatch     = prevDrive + b * numNeurons;
         // Activity is an extended buffer.
         float const *ABatch = A + b * (nx + rt + lt) * (ny + up + dn) * nf;

         int kex            = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
         float currentDrive = (1.0f - exp_tau) * (gSyn + selfInteract * ABatch[kex]);
         // Accumulate into VBatch with decay and momentum
         VBatch[k] = exp_tau * VBatch[k] + currentDrive + LCAMomentumRate * prevDriveBatch[k];
         // Update momentum buffer
         prevDriveBatch[k] = currentDrive;
      }
      else {
         float const *GSynInh = GSynHead + 1 * numNeurons * nbatch;
         int b                = kIndex / numNeurons;
         int k                = kIndex % numNeurons;

         float exp_tau             = (float)std::exp(-dtAdapt[b] / (double)tau);
         float *VBatch             = V + b * numNeurons;
         float const *GSynExcBatch = GSynExc + b * numNeurons;
         float const *GSynInhBatch = GSynInh + b * numNeurons;
         float const gSyn          = GSynExcBatch[k] - GSynInhBatch[k];
         float *prevDriveBatch     = prevDrive + b * numNeurons;
         // Activity is an extended buffer.
         float const *ABatch = A + b * (nx + rt + lt) * (ny + up + dn) * nf;

         int kex            = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
         float currentDrive = (1.0f - exp_tau) * (gSyn + selfInteract * ABatch[kex]);
         // Accumulate into VBatch with decay and momentum
         VBatch[k] = exp_tau * VBatch[k] + currentDrive + LCAMomentumRate * prevDriveBatch[k];
         // Update momentum buffer
         prevDriveBatch[k] = currentDrive;
      }
   }
}

int CudaUpdateMomentumLCAInternalState::do_run() {
   int currBlockSize = device->get_max_threads();
   // Ceil to get all weights
   int currGridSize = ceil(((float)params.numNeurons * params.nbatch) / currBlockSize);
   // Call function
   MomentumLCAInternalState_update_state<<<currGridSize, currBlockSize, 0, device->getStream()>>>(
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
         params.prevDrive,
         params.selfInteract,
         params.dtAdapt,
         params.tau,
         params.LCAMomentumRate,
         params.GSynHead,
         params.activity);
   return 0;
}

} // end namespace PVCuda
