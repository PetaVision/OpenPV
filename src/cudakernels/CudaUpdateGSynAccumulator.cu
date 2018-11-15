#include "CudaUpdateGSynAccumulator.hpp"
#include "conversions.hcu"

namespace PVCuda {

__global__ void GSynAccumulator_update_state(
      int const nbatch,
      int const numNeurons,
      int const numChannels,
      float const *channelCoefficients,
      float const *layerInput,
      float *accumulatedGSyn) {

   int kIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
   if (kIndex < numNeurons * nbatch) {
      float value = 0.0f;
      for (int ch = 0; ch < numChannels; ch++) {
         value += channelCoefficients[ch] * layerInput[ch * numNeurons * nbatch + kIndex];
      }
      accumulatedGSyn[kIndex] = value;
   }
}

int CudaUpdateGSynAccumulator::do_run() {
   int currBlockSize = device->get_max_threads();
   // Ceil to get all weights
   int currGridSize = ceil(((float)params.numNeurons * params.nbatch) / currBlockSize);
   // Call function
   GSynAccumulator_update_state<<<currGridSize, currBlockSize, 0, device->getStream()>>>(
         params.nbatch,
         params.numNeurons,
         params.numChannels,
         params.channelCoefficients,
         params.layerInput,
         params.accumulatedGSyn);
   return 0;
}

} // end namespace PVCuda
