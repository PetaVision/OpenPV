#define PV_RUN_ON_GPU

#include "GSynAccumulator.hpp"
#include "GSynAccumulator.kpp"

namespace PV {

void GSynAccumulator::runKernel() {
   PVLayerLoc const *loc            = getLayerLoc();
   long const numNeuronsAcrossBatch =
         (long)loc->nx * (long)loc->ny * (long)loc->nf * (long)loc->nbatch;
   float const *channelCoeffs = (float const *)mCudaChannelCoefficients->getPointer();
   float const *layerInput    = (float const *)mLayerInput->getCudaBuffer()->getPointer();
   float *bufferData          = (float *)getCudaBuffer()->getPointer();
   int currBlockSize          = mCudaDevice->get_max_threads();
   cudaStream_t cudaStream    = mCudaDevice->getStream();
   // Ceil to get all weights
   unsigned int currGridSize =
         (unsigned int)std::ceil(((float)numNeuronsAcrossBatch) / (float)currBlockSize);
   // Call function
   PVCuda::updateGSynAccumulatorOnGPU<<<currGridSize, currBlockSize, 0, cudaStream>>>(
         numNeuronsAcrossBatch, mNumInputChannels, channelCoeffs, layerInput, bufferData);
}

} // end namespace PV

#undef PV_RUN_ON_GPU
