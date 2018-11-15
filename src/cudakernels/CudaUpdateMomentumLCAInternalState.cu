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
      float *V,
      float *prevDrive,
      const bool selfInteract,
      double *dtAdapt,
      const float tau,
      const float LCAMomentumRate,
      float const *accumulatedGSyn,
      float const *A) {

   int kIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
   if (kIndex < numNeurons * nbatch) {
      int b = kIndex / numNeurons;
      int k = kIndex % numNeurons;

      float exp_tau                     = (float)std::exp(-dtAdapt[b] / (double)tau);
      float *VBatch                     = V + b * numNeurons;
      float const *accumulatedGSynBatch = accumulatedGSyn + b * numNeurons;
      float const gSyn                  = accumulatedGSynBatch[k]; // only one channel
      float *prevDriveBatch             = prevDrive + b * numNeurons;
      // Activity is an extended buffer.
      float const *ABatch = A + b * (nx + rt + lt) * (ny + up + dn) * nf;
      int kex             = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);

      float currentDrive = (1.0f - exp_tau) * (gSyn + selfInteract * ABatch[kex]);
      // Accumulate into VBatch with decay and momentum
      VBatch[k] = exp_tau * VBatch[k] + currentDrive + LCAMomentumRate * prevDriveBatch[k];
      // Update momentum buffer
      prevDriveBatch[k] = currentDrive;
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
         params.V,
         params.prevDrive,
         params.selfInteract,
         params.dtAdapt,
         params.tau,
         params.LCAMomentumRate,
         params.accumulatedGSyn,
         params.activity);
   return 0;
}

} // end namespace PVCuda
