/*
 * RecvPre.cu
 *
 *  Created on: Aug 5, 2014
 *      Author: Sheng Lundquist
 */

#ifndef CUDARECVPRE_HPP_
#define CUDARECVPRE_HPP_

#include "../arch/cuda/CudaBuffer.hpp"
#include "../arch/cuda/CudaKernel.hpp"
#include "../structures/SparseList.hpp"
//#include "../arch/cuda/Cuda3dFloatTextureBuffer.hpp"
//#include "../utils/conversions.h"
//#include "../layers/accumulate_functions.h"

namespace PVCuda {
#include <builtin_types.h>

typedef struct PVPatch_ {
   // float * __attribute__ ((aligned)) data;
   unsigned int offset;
   unsigned short nx, ny;
} Patch;

// Parameter structure
struct recv_pre_params {
   int nbatch;
   int numPreExt;
   int numPostRes;

   int nxp;
   int nyp;
   int nfp;

   int sy;
   int syw;
   float dt_factor;
   int sharedWeights;

   Patch *patches;
   size_t *gSynPatchStart;

   float *preData;
   float *weights;
   float *postGSyn;
   int *patch2datalookuptable;

   bool isSparse;
   long *numActive;
   PV::SparseList<float>::Entry *activeIndices;
};

class CudaRecvPre : public CudaKernel {
  public:
   CudaRecvPre(CudaDevice *inDevice);

   virtual ~CudaRecvPre();

   void setArgs(
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
         /* unsigned long * */ CudaBuffer *numActive,
         /* unsigned int* */ CudaBuffer *activeIndices);

   void set_dt_factor(float new_dt_factor) { params.dt_factor = new_dt_factor; }

  protected:
   // This is the function that should be overwritten in child classes
   virtual int do_run() override;

  private:
   void checkSharedMemSize(size_t sharedSize);

  private:
   recv_pre_params params;
   long *numActive;
}; // end class CudaRecvPre

} // end namespace PVCuda

#endif /* CLKERNEL_HPP_ */
