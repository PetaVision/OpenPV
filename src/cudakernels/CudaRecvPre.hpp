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
      int preNxExt;
      int preNyExt;
      int preNf;
      int postNxRes;
      int postNyRes;
      int postNf;

      int nxp;
      int nyp;
      int nfp;
      int groupXSize;
      int groupYSize;
      int localPreSizeX;
      int localPreSizeY;
      int localBufSizeX;
      int localBufSizeY;

      int sy;
      int syw;
      float dt_factor;
      int sharedWeights;

      PVPatch* patches;
      size_t* gSynPatchStart;

      long* postToPreActivity;
      float* preData;
      float* weights;
      float* postGSyn;
      int* patch2datalookuptable;

      bool isSparse;
      unsigned int numActive;
      unsigned int activeIndices;

      int preBufNum;
      int postBufNum;

   };


class CudaRecvPre : public CudaKernel {
public:
   CudaRecvPre(CudaDevice* inDevice);
   
   virtual ~CudaRecvPre();

   void setArgs(
      int preNxExt,
      int preNyExt,
      int preNf,
      int postNxRes,
      int postNyRes,
      int postNf,

      int nxp,
      int nyp,
      int nfp,
      int groupXSize,
      int groupYSize,
      int localPreSizeX,
      int localPreSizeY,
      int localBufSizeX,
      int localBufSizeY,

      int sy,
      int syw,
      float dt_factor,
      int sharedWeights,

      /* PVPatch* */ CudaBuffer* patches,
      /* size_t* */  CudaBuffer* gSynPatchStart,

      /* long* */    CudaBuffer* postToPreActivity,
      /* float* */   CudaBuffer* preData,
      /* float* */   CudaBuffer* weights,
      /* float* */   CudaBuffer* postGSyn,
      /* int* */     CudaBuffer* patch2datalookuptable

      //bool isSparse,
      //unsigned int numActive,
      //unsigned int * activeIndices
   );

   void setNumActive(unsigned int numActive){params.numActive = numActive;}

protected:
   //This is the function that should be overwritten in child classes
   virtual int do_run();

private:
   recv_pre_params params;
};

}

#endif /* CLKERNEL_HPP_ */
