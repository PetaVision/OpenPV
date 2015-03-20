/*
 * RecvPre.cu
 *
 *  Created on: Aug 5, 2014
 *      Author: Sheng Lundquist
 */

#ifndef CUDARECVPRE_HPP_
#define CUDARECVPRE_HPP_

#include "../arch/cuda/CudaKernel.hpp"
#include "../arch/cuda/CudaBuffer.hpp"
//#include "../arch/cuda/Cuda3dFloatTextureBuffer.hpp"
//#include "../utils/conversions.h"
//#include "../layers/accumulate_functions.h"

namespace PVCuda{
#include <builtin_types.h>

typedef struct PVPatch_ {
   // pvdata_t * __attribute__ ((aligned)) data;
   unsigned int offset;
   unsigned short nx, ny;
} PVPatch;

//Parameter structur
   struct recv_pre_params{
      int nxp;
      int nyp;
      int nfp;

      int sy;
      int syw;
      float dt_factor;
      int sharedWeights;

      PVPatch* patches;
      size_t* gSynPatchStart;

      float* preData;
      float* weights;
      float* postGSyn;
      int* patch2datalookuptable;

      bool isSparse;
      unsigned int * activeIndices;
      long numActive;
   };


class CudaRecvPre : public CudaKernel {
public:
   CudaRecvPre(CudaDevice* inDevice);
   
   virtual ~CudaRecvPre();

   void setArgs(
      int nxp,
      int nyp,
      int nfp,

      int sy,
      int syw,
      float dt_factor,
      int sharedWeights,

      /* PVPatch* */ CudaBuffer* patches,
      /* size_t* */  CudaBuffer* gSynPatchStart,

      /* float* */   CudaBuffer* preData,
      /* float* */   CudaBuffer* weights,
      /* float* */   CudaBuffer* postGSyn,
      /* int* */     CudaBuffer* patch2datalookuptable,

      bool isSparse,
      /* unsigned int* */ CudaBuffer* activeIndices
   );

   void set_dt_factor(float new_dt_factor) { params.dt_factor = new_dt_factor; }
   void set_numActive(long new_numActive) { params.numActive = new_numActive; }

protected:
   //This is the function that should be overwritten in child classes
   virtual int do_run();

private:
   recv_pre_params params;
};

}

#endif /* CLKERNEL_HPP_ */
