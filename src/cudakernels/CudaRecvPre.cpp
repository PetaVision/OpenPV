#include "CudaRecvPre.hpp"
#include "arch/cuda/cuda_util.hpp"
#include "utils/PVLog.hpp"

namespace PVCuda {

CudaRecvPre::CudaRecvPre(CudaDevice *inDevice) : CudaKernel(inDevice) {
   mKernelName = "CudaRecvPre";
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
      int channelCode,

      /* Patch* */ CudaBuffer *patches,
      /* size_t* */ CudaBuffer *gSynPatchStart,

      /* float* */ CudaBuffer *preData,
      /* float* */ CudaBuffer *weights,
      /* float* */ CudaBuffer *postGSyn,
      /* int* */ CudaBuffer *patch2datalookuptable,

      bool isSparse,
      /*unsigned long*/ CudaBuffer *numActive,
      /*unsigned int*/ CudaBuffer *activeIndices) {
    mParams.nbatch     = nbatch;
    mParams.numPreExt  = numPreExt;
    mParams.numPostRes = numPostRes;

    mParams.nxp = nxp;
    mParams.nyp = nyp;
    mParams.nfp = nfp;

    mParams.sy            = sy;
    mParams.syw           = syw;
    mParams.dt_factor     = dt_factor;
    mParams.sharedWeights = sharedWeights;
    mParams.channelCode   = channelCode;

    mParams.patches        = (PV::Patch *)patches->getPointer();
    mParams.gSynPatchStart = (size_t *)gSynPatchStart->getPointer();

    mParams.preData               = (float *)preData->getPointer();
    mParams.weights               = (float *)weights->getPointer();
    mParams.postGSyn              = (float *)postGSyn->getPointer();
    mParams.patch2datalookuptable = (int *)patch2datalookuptable->getPointer();

    mParams.isSparse = isSparse;
   if (activeIndices) {
      mParams.numActive     = (long *)numActive->getPointer();
      mParams.activeIndices = (PV::SparseList<float>::Entry *)activeIndices->getPointer();
   }
   else {
      mParams.activeIndices = NULL;
      mParams.numActive     = NULL;
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
