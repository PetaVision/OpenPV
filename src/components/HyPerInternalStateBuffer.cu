#define PV_RUN_ON_GPU

#include "HyPerInternalStateBuffer.hpp"
#include "HyPerInternalStateBuffer.kpp"

namespace PV {

void HyPerInternalStateBuffer::runKernel() {
   PVLayerLoc const *loc            = getLayerLoc();
   long const numNeuronsAcrossBatch =
         (long)loc->nx * (long)loc->ny * (long)loc->nf * (long)loc->nbatch;
   int const *channelIndices  = (int const *)mCudaChannelIndices->getPointer();
   float const *channelCoeffs = (float const *)mCudaChannelCoefficients->getPointer();
   float const *layerInput    = (float const *)mGSyn->getCudaBuffer()->getPointer();
   float *bufferData          = (float *)getCudaBuffer()->getPointer();
   int currBlockSize          = mCudaDevice->get_max_threads();
   cudaStream_t cudaStream    = mCudaDevice->getStream();
   // Ceil to get all weights
   int currGridSize = (int)ceil(((float)numNeuronsAcrossBatch) / (float)currBlockSize);
   // Call function
   PVCuda::updateHyPerInternalStateBufferOnGPU<<<currGridSize, currBlockSize, 0, cudaStream>>>(
         numNeuronsAcrossBatch,
         mNumChannelIndices,
         channelIndices,
         channelCoeffs,
         layerInput,
         bufferData);
}

} // end namespace PV

#undef PV_RUN_ON_GPU
