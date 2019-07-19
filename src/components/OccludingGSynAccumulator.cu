#define PV_RUN_ON_GPU

#include "OccludingGSynAccumulator.hpp"
#include "OccludingGSynAccumulator.kpp"

namespace PV {

void OccludingGSynAccumulator::runKernel() {
   PVLayerLoc const *loc           = getLayerLoc();
   int const numPixelsAcrossBatch  = loc->nx * loc->ny * loc->nbatch;
   float const *layerInput         = (float const *)mLayerInput->getCudaBuffer()->getPointer();
   float *bufferData               = (float *)getCudaBuffer()->getPointer();
   float *contribData              = (float *)mCudaContribData->getPointer();
   int currBlockSize               = mCudaDevice->get_max_threads();
   cudaStream_t cudaStream         = mCudaDevice->getStream();
   // Ceil to get all weights
   int currGridSize = (int)ceil(((float)numPixelsAcrossBatch) / currBlockSize);
   // Call function
   PVCuda::updateOccludingGSynAccumulatorOnGPU<<<currGridSize, currBlockSize, 0, cudaStream>>>(
          loc->nbatch, loc->nx, loc->ny, loc->nf, mNumChannels, layerInput, bufferData, contribData);
}

} // end namespace PV

#undef PV_RUN_ON_GPU
