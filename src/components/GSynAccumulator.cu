#include "GSynAccumulator.hpp"
#include "cudakernels/conversions.hcu"

namespace PVCuda {

__global__ void updateKernelGSynAccumulator(
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

} // end namespace PVCuda

namespace PV {

void GSynAccumulator::runKernel() {
   PVLayerLoc const *loc           = getLayerLoc();
   int const nx                    = loc->nx;
   int const ny                    = loc->ny;
   int const nf                    = loc->nf;
   int const numNeurons            = nx * ny * nf;
   int const nbatch                = loc->nbatch;
   int const numNeuronsAcrossBatch = numNeurons * nbatch;
   int currBlockSize               = mCudaDevice->get_max_threads();
   cudaStream_t cudaStream         = mCudaDevice->getStream();
   // Ceil to get all weights
   int currGridSize = (int)ceil(((float)numNeuronsAcrossBatch) / currBlockSize);
   // Call function
   PVCuda::updateKernelGSynAccumulator<<<currGridSize, currBlockSize, 0, cudaStream>>>(
         nbatch,
         numNeurons,
         mNumChannels,
         (float const *)mCudaChannelCoefficients->getPointer(),
         (float const *)mLayerInput->getCudaBuffer()->getPointer(),
         (float *)getCudaBuffer()->getPointer());
}

} // end namespace PV
