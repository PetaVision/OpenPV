#include "ANNActivityBuffer.hpp"

#define PV_RUN_ON_GPU
#include "ANNActivityBuffer.kpp"
#undef PV_RUN_ON_GPU

namespace PV {

void ANNActivityBuffer::runKernel() {
   PVLayerLoc const *loc           = getLayerLoc();
   int const numNeurons            = mInternalState->getBufferSize();
   int const nbatch                = loc->nbatch;
   int const numNeuronsAcrossBatch = numNeurons * nbatch;
   int currBlockSize               = mCudaDevice->get_max_threads();
   cudaStream_t cudaStream         = mCudaDevice->getStream();
   // Ceil to get all weights
   int currGridSize = (int)ceil((float)numNeuronsAcrossBatch / (float)currBlockSize);
   PVCuda::applyVerticesANNActivityBufferOnGPU<<<currGridSize, currBlockSize, 0, cudaStream>>>(
         nbatch,
         numNeurons,
         loc->nx,
         loc->ny,
         loc->nf,
         loc->halo.lt,
         loc->halo.rt,
         loc->halo.dn,
         loc->halo.up,
         mNumVertices,
         (float const *)mCudaVerticesV->getPointer(),
         (float const *)mCudaVerticesA->getPointer(),
         (float const *)mCudaSlopes->getPointer(),
         (float const *)mInternalState->getCudaBuffer()->getPointer(),
         (float *)getCudaBuffer()->getPointer());
}

} // end namespace PV
