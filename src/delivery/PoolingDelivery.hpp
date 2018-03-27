/*
 * PoolingDelivery.hpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#ifndef POOLINGDELIVERY_HPP_
#define POOLINGDELIVERY_HPP_

#include "components/ImpliedWeightsPair.hpp"
#include "components/PatchSize.hpp"
#include "delivery/BaseDelivery.hpp"
#include "layers/PoolingIndexLayer.hpp"
#ifdef PV_USE_CUDA
#include "cudakernels/CudaPoolingDeliverKernel.hpp"
#endif // PV_USE_CUDA

namespace PV {

/**
 * The delivery component for PoolingConns.
 */
class PoolingDelivery : public BaseDelivery {
  protected:
   /**
    * List of parameters needed from the PoolingDelivery class
    * @name PoolingDelivery Parameters
    * @{
    */

   /**
    * @brief pvpatchAccumulateType: Specifies the method to accumulate synaptic input
    * @details Possible choices are
    * - maxpooling: Takes the maximum value in the receptive field
    * - sumpooling: Takes the sum of all input values in the receptive field
    *        (equivalent to convolution where all weights are equal to one).
    * - avgpooling: Takes the average of all input values in the receptive field.
    *
    * This parameter is required.
    */
   virtual void ioParam_pvpatchAccumulateType(enum ParamsIOFlag ioFlag);

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

   // This doxygen comment may not be as edifying as it appears.
   /**
    * needPostIndexLayer: Set to true if a PostIndexLayer is needed; false otherwise.
    */
   void ioParam_needPostIndexLayer(enum ParamsIOFlag ioFlag);

   /**
    * If needPostIndexLayer is set, this parameter specifies the name of the PostIndexLayer.
    */
   void ioParam_postIndexLayerName(enum ParamsIOFlag ioFlag);
   /** @} */ // End of list of PoolingDelivery parameters.

  public:
   enum AccumulateType { UNDEFINED, MAXPOOLING, SUMPOOLING, AVGPOOLING };

   PoolingDelivery(char const *name, HyPerCol *hc);

   virtual ~PoolingDelivery();

   void setConnectionData(ConnectionData *connectionData);

   virtual void deliver() override;

   virtual bool isAllInputReady() override;

   AccumulateType getAccumulateType() const { return mAccumulateType; }

   PoolingIndexLayer *getPostIndexLayer() const { return mPostIndexLayer; }

   /**
    * Translates the input string into an accumulated type.
    * The parsing is case-insensitive, and the strings
    * "max pooling", "max_pooling", and "maxpooling" all translate
    * to MAXPOOLING. Sumpooling and avgpooling behave the same way.
    * If the string does not match any of the accumulation type,
    * the method returns UNDEFINED.
    */
   static AccumulateType parseAccumulateTypeString(char const *typestring);

  protected:
   PoolingDelivery();

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

   void initializeDeliverKernelArgs();

   void allocateThreadGSyn();

   void deliverPostsynapticPerspective();

   void deliverPresynapticPerspective();

   void clearGateIdxBuffer();

#ifdef PV_USE_CUDA
   void deliverGPU();
#endif // PV_USE_CUDA

   // Data members
  protected:
   AccumulateType mAccumulateType      = UNDEFINED;
   char *mPvpatchAccumulateTypeString  = nullptr;
   bool mUpdateGSynFromPostPerspective = false;

   PatchSize *mPatchSize            = nullptr;
   ImpliedWeightsPair *mWeightsPair = nullptr;

   bool mNeedPostIndexLayer           = false;
   char *mPostIndexLayerName          = nullptr;
   PoolingIndexLayer *mPostIndexLayer = nullptr;

   std::vector<std::vector<float>> mThreadGSyn;
   std::vector<std::vector<float>> mThreadGateIdxBuffer;
#ifdef PV_USE_CUDA
   PVCuda::CudaPoolingDeliverKernel *mRecvKernel = nullptr; // Cuda kernel for updating GSyn
#endif // PV_USE_CUDA

}; // end class PoolingDelivery

} // end namespace PV

#endif // POOLINGDELIVERY_HPP_
