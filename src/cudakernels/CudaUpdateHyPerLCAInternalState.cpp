#include "CudaUpdateHyPerLCAInternalState.hpp"

namespace PVCuda {

CudaUpdateHyPerLCAInternalState::CudaUpdateHyPerLCAInternalState(CudaDevice *inDevice)
      : CudaKernel(inDevice) {
   mKernelName = "HyPerLCAInternalState";
}

CudaUpdateHyPerLCAInternalState::~CudaUpdateHyPerLCAInternalState() {}

void CudaUpdateHyPerLCAInternalState::setArgs(
      int const nbatch,
      int const numNeurons,
      int const nx,
      int const ny,
      int const nf,
      int const lt,
      int const rt,
      int const dn,
      int const up,
      int const numChannels,

      CudaBuffer *V /*float*/,
      bool const selfInteract,
      CudaBuffer *dtAdapt /*double*/,
      float const tau,

      CudaBuffer *GSynHead /*float*/,
      CudaBuffer *activity /*float*/) {
   params.nbatch      = nbatch;
   params.numNeurons  = numNeurons;
   params.nx          = nx;
   params.ny          = ny;
   params.nf          = nf;
   params.lt          = lt;
   params.rt          = rt;
   params.dn          = dn;
   params.up          = up;
   params.numChannels = numChannels;

   params.V = (float *)V->getPointer();
   params.selfInteract = selfInteract;
   params.dtAdapt = (double *)dtAdapt->getPointer();
   params.tau     = tau;

   params.GSynHead = (float const *)GSynHead->getPointer();
   params.activity = (float const *)activity->getPointer();

   setArgsFlag();
}

} // end namespace PVCuda
