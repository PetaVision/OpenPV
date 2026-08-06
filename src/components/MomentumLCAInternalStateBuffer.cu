#define PV_RUN_ON_GPU

#include "MomentumLCAInternalStateBuffer.hpp"
#include "MomentumLCAInternalStateBuffer.kpp"

namespace PV {

void MomentumLCAInternalStateBuffer::runKernel() {
   PVLayerLoc const *loc  = getLayerLoc();
   long const numNeurons  = getBufferSize();
   int const nbatch       = loc->nbatch;

   double const *dtAdapt = (double const *)mCudaDtAdapt->getPointer();
   float const *GSyn     = (float const *)mGSyn->getCudaBuffer()->getPointer();
   float const *A        = (float const *)mActivity->getCudaBuffer()->getPointer();
   float *prevDrive      = (float *)mPrevDrive->getCudaBuffer()->getPointer();
   float *V              = (float *)getCudaBuffer()->getPointer();

   long const numNeuronsAcrossBatch = numNeurons * nbatch;
   int currBlockSize                = mCudaDevice->get_max_threads();
   cudaStream_t cudaStream          = mCudaDevice->getStream();
   // Ceil to get all weights
   unsigned int currGridSize =
         (unsigned int)std::ceil(((float)numNeuronsAcrossBatch) / (float)currBlockSize);
   // Call function
   PVCuda::updateMomentumLCAOnGPU<<<currGridSize, currBlockSize, 0, cudaStream>>>(
         nbatch,
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
         mLCAMomentumRate,
         dtAdapt,
         mScaledTimeConstantTau,
         GSyn,
         A,
         prevDrive,
         V);
}

} // end namespace PV

#undef PV_RUN_ON_GPU
