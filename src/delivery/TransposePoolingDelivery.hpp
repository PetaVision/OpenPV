/*
 * TransposePoolingDelivery.hpp
 *
 *  Created on: Jan 9, 2018
 *      Author: Pete Schultz
 */

#ifndef TRANSPOSEPOOLINGDELIVERY_HPP_
#define TRANSPOSEPOOLINGDELIVERY_HPP_

#include "components/DependentPatchSize.hpp"
#include "components/ImpliedWeightsPair.hpp"
#include "delivery/BaseDelivery.hpp"
#include "delivery/PoolingDelivery.hpp"
#include "layers/PoolingIndexLayer.hpp"
#ifdef PV_USE_CUDA
#include "cudakernels/CudaTransposePoolingDeliverKernel.hpp"
#endif // PV_USE_CUDA

namespace PV {

/**
 * The delivery component for PoolingConns.
 */
class TransposePoolingDelivery : public BaseDelivery {
  protected:
   /**
    * List of parameters needed from the TransposePoolingDelivery class
    * @name TransposePoolingDelivery Parameters
    * @{
    */

   /**
    * TransposePoolingDeliver does not read the receiveGpu flag, but uses the same
    * value as the original connection.
    */
   virtual void ioParam_receiveGpu(enum ParamsIOFlag ioFlag) override;

   /**
    * @brief updateGSynFromPostPerspective: Specifies if the connection should push from pre or pull
    * from post. This parameter is read if receiveGpu is false.
    * @details: If set to true, the connection loops over postsynaptic neurons, and each
    * post-neuron pulls from its receptive field. This avoids issues of collisions when
    * parallelizing, but is not able to take advantage of a sparse pre-layer.
    *
    * If false, the connection loops over presynaptic neurons, and each pre-neuron pushes to its
    * region of influence. This allows efficiency for sparse pre-layers, but requires extra memory
    * to manage potential collisions as multiple pre-neurons write to the same post-neuron.
    *
    * If the receiveGpu flag is set, the updateGSynFromPostPerspective is ignored, and the
    * cuDNN pooling routines are used.
    */
   virtual void ioParam_updateGSynFromPostPerspective(enum ParamsIOFlag ioFlag);

  public:
   TransposePoolingDelivery(char const *name, HyPerCol *hc);

   virtual ~TransposePoolingDelivery();

   void setConnectionData(ConnectionData *connectionData);

   virtual void deliver() override;

   virtual bool isAllInputReady() override;

  protected:
   TransposePoolingDelivery();

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

#ifdef PV_USE_CUDA
   virtual Response::Status
   setCudaDevice(std::shared_ptr<SetCudaDeviceMessage const> message) override;
#endif // PV_USE_CUDA

   virtual Response::Status allocateDataStructures() override;

#ifdef PV_USE_CUDA
   void initializeDeliverKernelArgs();
#endif // PV_USE_CUDA

   void allocateThreadGSyn();

   void deliverPostsynapticPerspective();

   void deliverPresynapticPerspective();

#ifdef PV_USE_CUDA
   void deliverGPU();
#endif // PV_USE_CUDA

   // Data members
  protected:
   PoolingDelivery::AccumulateType mAccumulateType = PoolingDelivery::UNDEFINED;
   bool mUpdateGSynFromPostPerspective             = false;

   DependentPatchSize *mPatchSize             = nullptr;
   ImpliedWeightsPair *mWeightsPair           = nullptr;
   PoolingIndexLayer *mOriginalPostIndexLayer = nullptr; // Used by deliverPresynapticPerspective
   HyPerLayer *mOriginalPreLayer              = nullptr; // Used by deliverGPU
   HyPerLayer *mOriginalPostLayer             = nullptr; // Used by deliverGPU

#ifdef PV_USE_CUDA
   PVCuda::CudaTransposePoolingDeliverKernel *mDeliverKernel = nullptr;
#endif // PV_USE_CUDA

   std::vector<std::vector<float>> mThreadGSyn;

}; // end class TransposePoolingDelivery

} // end namespace PV

#endif // TRANSPOSEPOOLINGDELIVERY_HPP_
