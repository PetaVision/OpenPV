/*
 * CudaPoolingRecvPost.hpp
 *
 *  Created on: Aug 2, 2016
 *      Author: pschultz
 */

#ifndef CUDAPOOLINGDELIVERKERNEL_HPP_
#define CUDAPOOLINGDELIVERKERNEL_HPP_

#include "arch/cuda/CudaKernel.hpp"
#include "include/PVLayerLoc.h"

namespace PVCuda {

class CudaPoolingDeliverKernel : public CudaKernel {
   struct Params {
      PVLayerLoc const * preLoc;
      PVLayerLoc const * postLoc;
      int diffX;
      int diffY;
      cudnnPoolingMode_t poolingMode;
      float multiplier;
      void * poolingDescriptor;
      void * dataStoreDescriptor;
      float * dataStore;
      void * gSynDescriptor;
      float * gSyn;
   };
public:
   CudaPoolingDeliverKernel(CudaDevice* inDevice);
   virtual ~CudaPoolingDeliverKernel();
   void setArgs(
         PVLayerLoc const * preLoc,
         PVLayerLoc const * postLoc,
         int nxpPost,
         int nypPost,
         cudnnPoolingMode_t poolingMode,
         int multiplier,
         CudaBuffer * inputBuffer,
         CudaBuffer * outputBuffer,
         int channel
         );
   static int calcStride(int preRestricted, int postRestricted);

protected:
   virtual int do_run() override;
   int calcBorderExcess(int preRestricted, int postRestricted, int border, int patchSizePostPerspective);
   int calcManyScale(int preRestricted, int postRestricted);

protected:
   PVLayerLoc const * mPreLoc = nullptr;
   PVLayerLoc const * mPostLoc = nullptr;
   int mBorderExcessX = 0;
   int mBorderExcessY = 0;
   cudnnPoolingMode_t mPoolingMode = CUDNN_POOLING_MAX;
   float mMultiplier = 1.0f;
   cudnnPoolingDescriptor_t mPoolingDescriptor = nullptr;
   cudnnTensorDescriptor_t mDataStoreDescriptor = nullptr;
   float * mDataStore = nullptr;
   CudaBuffer * mCudnnDataStore = nullptr;

   cudnnTensorDescriptor_t mGSynDescriptor = nullptr;
   float * mGSyn = nullptr;
   CudaBuffer * mCudnnGSyn = nullptr;
};

} /* namespace PVCuda */

#endif /* CUDAPOOLINGDELIVERKERNEL_HPP_ */
