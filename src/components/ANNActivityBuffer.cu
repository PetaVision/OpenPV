#include "ANNActivityBuffer.hpp"
#include "cudakernels/conversions.hcu"

namespace PVCuda {

__global__ void updateKernelANNActivityBuffer(
      int const nbatch,
      int const numNeurons,
      int const nx,
      int const ny,
      int const nf,
      int const lt,
      int const rt,
      int const dn,
      int const up,
      float const *V,
      int const numVertices,
      float const *verticesV,
      float const *verticesA,
      float const *slopes,
      float *A) {

   int kbatch = (blockIdx.x * blockDim.x) + threadIdx.x;
   if (kbatch < numNeurons * nbatch) {
      int last            = numVertices - 1;
      int b               = kbatch / numNeurons;
      int k               = kbatch % numNeurons;
      float const *VBatch = V + b * numNeurons;
      float *ABatch       = A + b * (nx + lt + rt) * (ny + up + dn) * nf;
      int kex             = kIndexExtended(k, nx, ny, nf, lt, rt, dn, up);
      int v;
      float potential = VBatch[k];
      float activity  = 0.0f;

      if (potential < verticesV[0]) {
         activity = verticesA[0] + slopes[0] * (potential - verticesV[0]);
      }
      else if (potential >= verticesV[last]) {
         activity = verticesA[last] + slopes[numVertices] * (potential - verticesV[last]);
      }
      else {
         for (v = 0; v < last; v++) {
            if (potential < verticesV[v]) {
               break; // makes the jumps continuous from the right.  TODO: allow user control over
               // value at jump
            }
            if (potential == verticesV[v]) {
               activity = verticesA[v];
            }
            else if (potential > verticesV[v] && potential < verticesV[v + 1]) {
               activity = verticesA[v] + slopes[v + 1] * (potential - verticesV[v]);
            }
         }
      }
      ABatch[kex] = activity;
   }
}

} // end namespace PVCuda

namespace PV {

void ANNActivityBuffer::runKernel() {
   PVLayerLoc const *loc           = getLayerLoc();
   int const nx                    = loc->nx;
   int const ny                    = loc->ny;
   int const nf                    = loc->nf;
   int const numNeurons            = nx * ny * nf;
   int const nbatch                = loc->nbatch;
   int const lt                    = loc->halo.lt;
   int const rt                    = loc->halo.rt;
   int const dn                    = loc->halo.dn;
   int const up                    = loc->halo.up;
   int const numNeuronsAcrossBatch = numNeurons * nbatch;
   int currBlockSize               = mCudaDevice->get_max_threads();
   cudaStream_t cudaStream         = mCudaDevice->getStream();
   // Ceil to get all weights
   int currGridSize = (int)ceil((float)numNeuronsAcrossBatch / (float)currBlockSize);
   PVCuda::updateKernelANNActivityBuffer<<<currGridSize, currBlockSize, 0, cudaStream>>>(
         nbatch,
         numNeurons,
         nx,
         ny,
         nf,
         lt,
         rt,
         dn,
         up,
         (float const *)mInternalState->getCudaBuffer()->getPointer(),
         mNumVertices,
         (float const *)mCudaVerticesV->getPointer(),
         (float const *)mCudaVerticesA->getPointer(),
         (float const *)mCudaSlopes->getPointer(),
         (float *)getCudaBuffer()->getPointer());
}

} // end namespace PV
