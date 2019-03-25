#include "MomentumLCAInternalStateBuffer.hpp"

#define PV_RUN_ON_GPU
#include "MomentumLCAInternalStateBuffer.kpp"
#undef PV_RUN_ON_GPU

namespace PV {

void MomentumLCAInternalStateBuffer::runKernel() {
   PVLayerLoc const *loc = getLayerLoc();
   int const numNeurons  = getBufferSize();
   int const nbatch      = loc->nbatch;

   double const *dtAdapt        = (double const *)mCudaDtAdapt->getPointer();
   float const *accumulatedGSyn = (float const *)mAccumulatedGSyn->getCudaBuffer()->getPointer();
   float const *A               = (float const *)mActivity->getCudaBuffer()->getPointer();
   float *prevDrive             = (float *)mPrevDrive->getCudaBuffer()->getPointer();
   float *V                     = (float *)getCudaBuffer()->getPointer();

   int const numNeuronsAcrossBatch = numNeurons * nbatch;
   int currBlockSize               = mCudaDevice->get_max_threads();
   cudaStream_t cudaStream         = mCudaDevice->getStream();
   // Ceil to get all weights
   int currGridSize = (int)ceil(((float)numNeuronsAcrossBatch) / currBlockSize);
   // Call function
   PVCuda::updateMomentumLCAOnGPU<<<currGridSize, currBlockSize, 0, cudaStream>>>(
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
         mLCAMomentumRate,
         dtAdapt,
         mScaledTimeConstantTau,
         accumulatedGSyn,
         A,
         prevDrive,
         V);
}

} // end namespace PV
