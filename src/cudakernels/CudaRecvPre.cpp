#include "CudaRecvPre.hpp"
#include "arch/cuda/cuda_util.hpp"
#include "conversions.hcu"
#include "utils/PVLog.hpp"

namespace PVCuda {

CudaRecvPre::CudaRecvPre(CudaDevice *inDevice) : CudaKernel(inDevice) {
   kernelName = "CudaRecvPre";
   numActive  = nullptr;
}

CudaRecvPre::~CudaRecvPre() {}

void CudaRecvPre::setArgs(
      int nbatch,
      int numPreExt,
      int numPostRes,
      int nxp,
      int nyp,
      int nfp,

      int sy,
      int syw,
      float dt_factor,
      int sharedWeights,

      /* Patch* */ CudaBuffer *patches,
      /* size_t* */ CudaBuffer *gSynPatchStart,

      /* float* */ CudaBuffer *preData,
      /* float* */ CudaBuffer *weights,
      /* float* */ CudaBuffer *postGSyn,
      /* int* */ CudaBuffer *patch2datalookuptable,

      bool isSparse,
      /*unsigned long*/ CudaBuffer *numActive,
      /*unsigned int*/ CudaBuffer *activeIndices) {
   params.nbatch     = nbatch;
   params.numPreExt  = numPreExt;
   params.numPostRes = numPostRes;

   params.nxp = nxp;
   params.nyp = nyp;
   params.nfp = nfp;

   params.sy            = sy;
   params.syw           = syw;
   params.dt_factor     = dt_factor;
   params.sharedWeights = sharedWeights;

   params.patches        = (Patch *)patches->getPointer();
   params.gSynPatchStart = (size_t *)gSynPatchStart->getPointer();

   params.preData               = (float *)preData->getPointer();
   params.weights               = (float *)weights->getPointer();
   params.postGSyn              = (float *)postGSyn->getPointer();
   params.patch2datalookuptable = (int *)patch2datalookuptable->getPointer();

   params.isSparse = isSparse;
   if (activeIndices) {
      params.numActive     = (long *)numActive->getPointer();
      params.activeIndices = (PV::SparseList<float>::Entry *)activeIndices->getPointer();
   }
   else {
      params.activeIndices = NULL;
      params.numActive     = NULL;
   }

   setArgsFlag();
}

void CudaRecvPre::checkSharedMemSize(size_t sharedSize) {
   if (sharedSize > device->get_local_mem()) {
      ErrorLog().printf(
            "run: given shared memory size of %zu is bigger than allowed shared memory size of "
            "%zu\n",
            sharedSize,
            device->get_local_mem());
   }
}

} // end namespace PVCuda
