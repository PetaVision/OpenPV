#include "CudaUpdateANNActivity.hpp"
#include "conversions.hcu"

namespace PVCuda {

__global__ void ANNActivity_update_state(
      const int nbatch,
      const int numNeurons,
      const int nx,
      const int ny,
      const int nf,
      const int lt,
      const int rt,
      const int dn,
      const int up,
      float const *V,
      const int numVertices,
      const float *verticesV,
      const float *verticesA,
      const float *slopes,
      float *A) {

   int kIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
   if (kIndex < numNeurons * nbatch) {
      int last            = numVertices - 1;
      int b               = kIndex / numNeurons;
      int k               = kIndex % numNeurons;
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

int CudaUpdateANNActivity::do_run() {
   int currBlockSize = device->get_max_threads();
   // Ceil to get all weights
   int currGridSize = ceil(((float)mParams.numNeurons * mParams.nbatch) / currBlockSize);
   // Call function
   ANNActivity_update_state<<<currGridSize, currBlockSize, 0, device->getStream()>>>(
         mParams.nbatch,
         mParams.numNeurons,
         mParams.nx,
         mParams.ny,
         mParams.nf,
         mParams.lt,
         mParams.rt,
         mParams.dn,
         mParams.up,
         mParams.internalState,
         mParams.numVertices,
         mParams.verticesV,
         mParams.verticesA,
         mParams.slopes,
         mParams.activity);
   return 0;
}

} // end namespace PVCuda
