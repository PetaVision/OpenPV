#include "CudaUpdateHyPerLCALayer.hpp"
#include "../arch/cuda/cuda_util.hpp"
#include "conversions.hcu"



namespace PVCuda{
//Include update state functions with cuda flag on 
#include "../layers/updateStateFunctions.h"

//The actual wrapper kernel code thats calling updatestatefunctions
__global__
void HyPerLCALayer_update_state(
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

    float * V,
    const float Vth,
    const float AMax,
    const float AMin,
    const float AShift,
    const float VWidth,
    const bool selfInteract,
    double * dtAdapt,
    const float tau,
    float * GSynHead,
    float * activity)
{

   if((blockIdx.x * blockDim.x) + threadIdx.x < numNeurons*nbatch){
      updateV_HyPerLCALayer(nbatch, numNeurons, V, GSynHead, activity,
            AMax, AMin, Vth, AShift, VWidth, dtAdapt, tau, selfInteract, nx, ny, nf, lt, rt, dn, up, numChannels);
   }
}


CudaUpdateHyPerLCALayer::CudaUpdateHyPerLCALayer(CudaDevice* inDevice):CudaKernel(inDevice){
}

CudaUpdateHyPerLCALayer::~CudaUpdateHyPerLCALayer(){
}

void CudaUpdateHyPerLCALayer::setArgs(
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

      /* float* */ CudaBuffer* V,

      const float Vth,
      const float AMax,
      const float AMin,
      const float AShift,
      const float VWidth,
      const bool selfInteract,
      /* double* */ CudaBuffer* dtAdapt,
      const float tau,

      /* float* */ CudaBuffer* GSynHead,
      /* float* */ CudaBuffer* activity
   ){
   params.nbatch = nbatch;
   params.numNeurons = numNeurons;
   params.nx = nx;
   params.ny = ny;
   params.nf = nf;
   params.lt = lt;
   params.rt = rt;
   params.dn = dn;
   params.up = up;
   params.numChannels = numChannels;

   params.V = (float*) V->getPointer();

   params.Vth = Vth;
   params.AMax = AMax;
   params.AMin = AMin;
   params.AShift = AShift;
   params.VWidth = VWidth;
   params.selfInteract = selfInteract;
   params.dtAdapt = (double*) dtAdapt->getPointer();
   params.tau = tau;

   params.GSynHead = (float*) GSynHead->getPointer();
   params.activity = (float*) activity->getPointer();

   setArgsFlag();
}

int CudaUpdateHyPerLCALayer::do_run(){
   int currBlockSize = device->get_max_threads();
   //Ceil to get all weights
   int currGridSize = ceil(((float)params.numNeurons * params.nbatch)/currBlockSize);
   //Call function
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
   params.Vth,
   params.AMax,
   params.AMin,
   params.AShift,
   params.VWidth,
   params.selfInteract,
   params.dtAdapt,
   params.tau,
   params.GSynHead,
   params.activity);
   handleCallError("HyPerLCALayer Update run");
   return 0;
}

}
