#include "CudaUpdateMomentumLCAInternalState.hpp"

namespace PVCuda {

CudaUpdateMomentumLCAInternalState::CudaUpdateMomentumLCAInternalState(CudaDevice *inDevice)
      : CudaKernel(inDevice) {
   mKernelName = "MomentumLCAInternalState";
}

CudaUpdateMomentumLCAInternalState::~CudaUpdateMomentumLCAInternalState() {}

void CudaUpdateMomentumLCAInternalState::setArgs(
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

      CudaBuffer *V /* float* */,
      CudaBuffer *prevDrive /* float* */,
      const bool selfInteract,
      CudaBuffer *dtAdapt /* double* */,
      const float tau,
      const float LCAMomentumRate,

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

   params.V               = (float *)V->getPointer();
   params.prevDrive       = (float *)prevDrive->getPointer();
   params.selfInteract    = selfInteract;
   params.dtAdapt         = (double *)dtAdapt->getPointer();
   params.tau             = tau;
   params.LCAMomentumRate = LCAMomentumRate;

   params.GSynHead = (float *)GSynHead->getPointer();
   params.activity = (float *)activity->getPointer();

   setArgsFlag();
}

} // end namespace PVCuda
