#include "CudaUpdateANNActivity.hpp"

namespace PVCuda {

CudaUpdateANNActivity::CudaUpdateANNActivity(CudaDevice *inDevice) : CudaKernel(inDevice) {
   mKernelName = "ANNActivity";
}

CudaUpdateANNActivity::~CudaUpdateANNActivity() {}

void CudaUpdateANNActivity::setArgs(
      int const nbatch,
      int const numNeurons,
      int const nx,
      int const ny,
      int const nf,
      int const lt,
      int const rt,
      int const dn,
      int const up,

      CudaBuffer *V /*float*/,
      int const numVertices,
      CudaBuffer *verticesV /*float*/,
      CudaBuffer *verticesA /*float*/,
      CudaBuffer *slopes /*float*/,
      CudaBuffer *A /*float*/) {
   mParams.nbatch     = nbatch;
   mParams.numNeurons = numNeurons;
   mParams.nx         = nx;
   mParams.ny         = ny;
   mParams.nf         = nf;
   mParams.lt         = lt;
   mParams.rt         = rt;
   mParams.dn         = dn;
   mParams.up         = up;

   mParams.internalState = (float const *)V->getPointer();
   mParams.numVertices   = numVertices;
   mParams.verticesV     = (float const *)verticesV->getPointer();
   mParams.verticesA     = (float const *)verticesA->getPointer();
   mParams.slopes        = (float const *)slopes->getPointer();
   mParams.activity      = (float *)A->getPointer();

   setArgsFlag();
}

} // end namespace PVCuda
