#define PV_RUN_ON_GPU

#include "ISTAInternalStateBuffer.hpp"
#include "ISTAInternalStateBuffer.kpp"

namespace PV {

void ISTAInternalStateBuffer::runKernel() {
   PVLayerLoc const *loc           = getLayerLoc();
   int const numNeurons            = loc->nx * loc->ny * loc->nf;
   int const numNeuronsAcrossBatch = numNeurons * loc->nbatch;
   int currBlockSize               = mCudaDevice->get_max_threads();
   cudaStream_t cudaStream         = mCudaDevice->getStream();
   // Ceil to get all weights
   int currGridSize = (int)ceil(((float)numNeuronsAcrossBatch) / currBlockSize);
   // Call function
   PVCuda::updateISTAInternalStateBufferOnGPU<<<currGridSize, currBlockSize, 0, cudaStream>>>(
         loc->nbatch,
         numNeurons,
         loc->nx,
         loc->ny,
         loc->nf,
         loc->halo.lt,
         loc->halo.rt,
         loc->halo.dn,
         loc->halo.up,
         mActivity->getVThresh(),
         (double const *)mCudaDtAdapt->getPointer(),
         mScaledTimeConstantTau,
         (float const *)mAccumulatedGSyn->getCudaBuffer()->getPointer(),
         (float const *)mActivity->getCudaBuffer()->getPointer(),
         (float *)getCudaBuffer()->getPointer());
}

} // end namespace PVCuda

#undef PV_RUN_ON_GPU
