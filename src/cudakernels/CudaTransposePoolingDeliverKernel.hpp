/*
 * CudaTransposePoolingDeliverKernel.hpp
 *
 *  Created on: Aug 16, 2016
 *      Author: pschultz
 */

#ifndef CUDATRANSPOSEPOOLINGDELIVERKERNEL_HPP_
#define CUDATRANSPOSEPOOLINGDELIVERKERNEL_HPP_

#include <cudakernels/CudaPoolingDeliverKernel.hpp>

namespace PVCuda {

class CudaTransposePoolingDeliverKernel: public CudaKernel {
public:
   CudaTransposePoolingDeliverKernel(CudaDevice * inDevice) : CudaKernel(inDevice) {}
   virtual ~CudaTransposePoolingDeliverKernel();
   void setArgs(
          PVLayerLoc const * preLoc,
          PVLayerLoc const * postLoc,
          PVLayerLoc const * origConnPreLoc,
          PVLayerLoc const * origConnPostLoc,
          int nxpPost,
          int nypPost,
          cudnnPoolingMode_t poolingMode,
          int multiplier,
          CudaBuffer * dataStoreBuffer,
          CudaBuffer * gSynBuffer,
          CudaBuffer * origConnDataStoreBuffer,
          CudaBuffer * origConnGSynBuffer,
          int channel
          );

protected:
   virtual int do_run() override;
   int calcBorderExcess(int preRestricted, int postRestricted, int border, int patchSizePostPerspective);
   int calcManyScale(int preRestricted, int postRestricted);
   int calcStride(int preRestricted, int postRestricted);

// Data members
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

   PVLayerLoc const * mOrigConnPreLoc = nullptr;
   PVLayerLoc const * mOrigConnPostLoc = nullptr;
   int mOrigConnBorderExcessX = 0;
   int mOrigConnBorderExcessY = 0;

   cudnnTensorDescriptor_t mOrigConnDataStoreDescriptor = nullptr;
   float * mOrigConnDataStore = nullptr;
   CudaBuffer * mCudnnOrigConnDataStore = nullptr;

   cudnnTensorDescriptor_t mOrigConnGSynDescriptor = nullptr;
   float * mOrigConnGSyn = nullptr;
   CudaBuffer * mCudnnOrigConnGSyn = nullptr;
}; // class CudaTransposePoolingDeliverKernel

} /* namespace PVCuda */

#endif /* CUDATRANSPOSEPOOLINGDELIVERKERNEL_HPP_ */
