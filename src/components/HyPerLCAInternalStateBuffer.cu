#define PV_RUN_ON_GPU

#include "HyPerLCAInternalStateBuffer.hpp"
#include "HyPerLCAInternalStateBuffer.kpp"

namespace PV {

void HyPerLCAInternalStateBuffer::runKernel() {
   PVLayerLoc const *loc            = getLayerLoc();
   long const numNeurons            = getBufferSize();
   long const numNeuronsAcrossBatch = getBufferSizeAcrossBatch();
   int currBlockSize                = mCudaDevice->get_max_threads();
   cudaStream_t cudaStream          = mCudaDevice->getStream();
   // Ceil to get all weights
   unsigned int currGridSize =
         (unsigned int)std::ceil(((float)numNeuronsAcrossBatch) / (float)currBlockSize);
   // Call function
   PVCuda::updateHyPerLCAOnGPU<<<currGridSize, currBlockSize, 0, cudaStream>>>(
         loc->nbatch,
         numNeurons,
         mNumChannelIndices,
         (int const *)mCudaChannelIndices->getPointer(),
         (float const *)mCudaChannelCoefficients->getPointer(),
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
         (float const *)mGSyn->getCudaBuffer()->getPointer(),
         (float const *)mActivity->getCudaBuffer()->getPointer(),
         (float *)getCudaBuffer()->getPointer());
}

} // end namespace PV

#undef PV_RUN_ON_GPU
