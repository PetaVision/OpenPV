#include "GSynAccumulator.hpp"

#define PV_RUN_ON_GPU
#include "GSynAccumulator.kpp"
#undef PV_RUN_ON_GPU

namespace PV {

void GSynAccumulator::runKernel() {
   PVLayerLoc const *loc           = getLayerLoc();
   int const numNeuronsAcrossBatch = loc->nx * loc->ny * loc->nf * loc->nbatch;
   float const *channelCoeffs      = (float const *)mCudaChannelCoefficients->getPointer();
   float const *layerInput         = (float const *)mLayerInput->getCudaBuffer()->getPointer();
   float *bufferData               = (float *)getCudaBuffer()->getPointer();
   int currBlockSize               = mCudaDevice->get_max_threads();
   cudaStream_t cudaStream         = mCudaDevice->getStream();
   // Ceil to get all weights
   int currGridSize = (int)ceil(((float)numNeuronsAcrossBatch) / currBlockSize);
   // Call function
   PVCuda::updateGSynAccumulatorOnGPU<<<currGridSize, currBlockSize, 0, cudaStream>>>(
         numNeuronsAcrossBatch, mNumInputChannels, channelCoeffs, layerInput, bufferData);
}

} // end namespace PV
