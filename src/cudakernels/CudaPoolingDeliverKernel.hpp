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

protected:
   virtual int do_run() override;
   int calcBorderExcess(int preRestricted, int postRestricted, int border, int patchSizePostPerspective);
   int calcManyScale(int preRestricted, int postRestricted);
   int calcStride(int preRestricted, int postRestricted);

protected:
   Params params;
   CudaBuffer * cudnnDataStore = nullptr;
   CudaBuffer * cudnnGSyn = nullptr;
};

} /* namespace PVCuda */

#endif /* CUDAPOOLINGDELIVERKERNEL_HPP_ */
