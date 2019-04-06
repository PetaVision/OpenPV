#define PV_RUN_ON_GPU

#include "HyPerLCAInternalStateBuffer.hpp"
#include "HyPerLCAInternalStateBuffer.kpp"

namespace PV {

void HyPerLCAInternalStateBuffer::runKernel() {
   PVLayerLoc const *loc           = getLayerLoc();
   int const numNeurons            = getBufferSize();
   int const nbatch                = loc->nbatch;
   int const numNeuronsAcrossBatch = numNeurons * nbatch;
   int currBlockSize               = mCudaDevice->get_max_threads();
   cudaStream_t cudaStream         = mCudaDevice->getStream();
   // Ceil to get all weights
   int currGridSize = (int)ceil(((float)numNeuronsAcrossBatch) / currBlockSize);
   // Call function
   PVCuda::updateHyPerLCAOnGPU<<<currGridSize, currBlockSize, 0, cudaStream>>>(
         nbatch,
         numNeurons,
         loc->nx,
         loc->ny,
         loc->nf,
         loc->halo.lt,
         loc->halo.rt,
         loc->halo.dn,
         loc->halo.up,
         mSelfInteract,
         (double const *)mCudaDtAdapt->getPointer(),
         mScaledTimeConstantTau,
         (float const *)mAccumulatedGSyn->getCudaBuffer()->getPointer(),
         (float const *)mActivity->getCudaBuffer()->getPointer(),
         (float *)getCudaBuffer()->getPointer());
}

} // end namespace PV

#undef PV_RUN_ON_GPU
