#include "CudaUpdateStateFunctions.hpp"
#include "conversions.hcu"

namespace PVCuda {
// Include update state functions with cuda flag on
#include "../layers/updateStateFunctions.h"

// The actual wrapper kernel code thats calling updatestatefunctions
__global__ void HyPerLCALayer_update_state(
      const int nbatch,
      const int numNeurons,
      const int nx,
      const int ny,
      const int nf,
      const int lt,
      const int rt,
      const int dn,
      const int up,
      const int numChannels,
      float *V,
      const int numVertices,
      float *verticesV,
      float *verticesA,
      float *slopes,
      const bool selfInteract,
      double *dtAdapt,
      const float tau,
      float *GSynHead,
      float *activity) {

   if ((blockIdx.x * blockDim.x) + threadIdx.x < numNeurons * nbatch) {
      updateV_HyPerLCALayer(
            nbatch,
            numNeurons,
            numChannels,
            V,
            GSynHead,
            activity,
            numVertices,
            verticesV,
            verticesA,
            slopes,
            dtAdapt,
            tau,
            selfInteract,
            nx,
            ny,
            nf,
            lt,
            rt,
            dn,
            up);
   }
}

__global__ void MomentumLCALayer_update_state(
      const int nbatch,
      const int numNeurons,
      const int nx,
      const int ny,
      const int nf,
      const int lt,
      const int rt,
      const int dn,
      const int up,
      const int numChannels,
      float *V,
      float *prevDrive,
      const int numVertices,
      float *verticesV,
      float *verticesA,
      float *slopes,
      const bool selfInteract,
      double *dtAdapt,
      const float tau,
      const float LCAMomentumRate,
      float *GSynHead,
      float *activity) {

   if ((blockIdx.x * blockDim.x) + threadIdx.x < numNeurons * nbatch) {
      updateV_MomentumLCALayer(
            nbatch,
            numNeurons,
            numChannels,
            V,
            GSynHead,
            activity,
            prevDrive,
            numVertices,
            verticesV,
            verticesA,
            slopes,
            dtAdapt,
            tau,
            LCAMomentumRate,
            selfInteract,
            nx,
            ny,
            nf,
            lt,
            rt,
            dn,
            up);
   }
}

__global__ void ISTALayer_update_state(
      const int nbatch,
      const int numNeurons,
      const int nx,
      const int ny,
      const int nf,
      const int lt,
      const int rt,
      const int dn,
      const int up,
      const int numChannels,
      float *V,
      const float Vth,
      double *dtAdapt,
      const float tau,
      float *GSynHead,
      float *activity) {
   if ((blockIdx.x * blockDim.x) + threadIdx.x < numNeurons * nbatch) {
      updateV_ISTALayer(
            nbatch,
            numNeurons,
            V,
            GSynHead,
            activity,
            Vth,
            dtAdapt,
            tau,
            nx,
            ny,
            nf,
            lt,
            rt,
            dn,
            up,
            numChannels);
   }
}

int CudaUpdateHyPerLCALayer::do_run() {
   int currBlockSize = device->get_max_threads();
   // Ceil to get all weights
   int currGridSize = ceil(((float)params.numNeurons * params.nbatch) / currBlockSize);
   // Call function
   HyPerLCALayer_update_state<<<currGridSize, currBlockSize, 0, device->getStream()>>>(
         params.nbatch,
         params.numNeurons,
         params.nx,
         params.ny,
         params.nf,
         params.lt,
         params.rt,
         params.dn,
         params.up,
         params.numChannels,
         params.V,
         params.numVertices,
         params.verticesV,
         params.verticesA,
         params.slopes,
         params.selfInteract,
         params.dtAdapt,
         params.tau,
         params.GSynHead,
         params.activity);
   return 0;
}

int CudaUpdateMomentumLCALayer::do_run() {
   int currBlockSize = device->get_max_threads();
   // Ceil to get all weights
   int currGridSize = ceil(((float)params.numNeurons * params.nbatch) / currBlockSize);
   // Call function
   MomentumLCALayer_update_state<<<currGridSize, currBlockSize, 0, device->getStream()>>>(
         params.nbatch,
         params.numNeurons,
         params.nx,
         params.ny,
         params.nf,
         params.lt,
         params.rt,
         params.dn,
         params.up,
         params.numChannels,
         params.V,
         params.prevDrive,
         params.numVertices,
         params.verticesV,
         params.verticesA,
         params.slopes,
         params.selfInteract,
         params.dtAdapt,
         params.tau,
         params.LCAMomentumRate,
         params.GSynHead,
         params.activity);
   return 0;
}

int CudaUpdateISTALayer::do_run() {
   int currBlockSize = device->get_max_threads();
   // Ceil to get all weights
   int currGridSize = ceil(((float)params.numNeurons * params.nbatch) / currBlockSize);
   // Call function
   ISTALayer_update_state<<<currGridSize, currBlockSize, 0, device->getStream()>>>(
         params.nbatch,
         params.numNeurons,
         params.nx,
         params.ny,
         params.nf,
         params.lt,
         params.rt,
         params.dn,
         params.up,
         params.numChannels,
         params.V,
         params.Vth,
         params.dtAdapt,
         params.tau,
         params.GSynHead,
         params.activity);
   return 0;
}

} // end namespace PVCuda
