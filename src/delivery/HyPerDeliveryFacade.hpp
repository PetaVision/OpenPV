/*
 * HyPerDeliveryFacade.hpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#ifndef HYPERDELIVERYFACADE_HPP_
#define HYPERDELIVERYFACADE_HPP_

#include "delivery/BaseDelivery.hpp"
#include "delivery/HyPerDelivery.hpp"

namespace PV {

/**
 * The delivery component for HyPerConns. It reads the parameters pvpatchAccumulateType,
 * updateGSynFromPostPerspective, and receiveGpu. Note that the choice of pvpatchAccumulateType
 * affects the result of the convolution. The other parameters do not (except for round-off
 * errors induced by parallelism); they only change how the calculation is performed.
 */
class HyPerDeliveryFacade : public BaseDelivery {
  protected:
   /**
    * List of parameters needed from the HyPerDeliveryFacade class
    * @name HyPerDeliveryFacade Parameters
    * @{
    */

   /**
    * @brief pvpatchAccumulateType: Specifies the method to accumulate synaptic input
    * @details Possible choices are
    * - convolve: Accumulates through convolution
    * - stochastic: Accumulates through stochastic release
    *
    * Defaults to convolve.
    */
   virtual void ioParam_accumulateType(enum ParamsIOFlag ioFlag);

   /**
    * @brief updateGSynFromPostPerspective: Specifies if the connection should push from pre or pull
    * from post.
    * @details: If set to true, the connection loops over postsynaptic neurons, and each
    * post-neuron pulls from its receptive field. This avoids issues of collisions when
    * parallelizing, but is not able to take advantage of a sparse pre-layer.
    *
    * If false, the connection loops over presynaptic neurons, and each pre-neuron pushes to its
    * region of influence. This allows efficiency for sparse pre-layers, but requires extra memory
    * to manage potential collisions as multiple pre-neurons write to the same post-neuron.
    */
   virtual void ioParam_updateGSynFromPostPerspective(enum ParamsIOFlag ioFlag);
   /** @} */ // End of list of HyPerDeliveryFacade parameters.

  public:
   HyPerDeliveryFacade(char const *name, HyPerCol *hc);

   virtual ~HyPerDeliveryFacade();

   virtual void deliver() override;

   virtual void deliverUnitInput(float *recvBuffer) override;

   virtual bool isAllInputReady() override;

   HyPerDelivery::AccumulateType getAccumulateType() const { return mAccumulateType; }

   bool getUpdateGSynFromPostPerspective() const { return mUpdateGSynFromPostPerspective; }

   bool getConvertRateToSpikeCount() const { return mConvertRateToSpikeCount; }

  protected:
   HyPerDeliveryFacade();

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   void createDeliveryIntern();

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

#ifdef PV_USE_CUDA
   virtual Response::Status
   setCudaDevice(std::shared_ptr<SetCudaDeviceMessage const> message) override;
#endif // PV_USE_CUDA

   virtual Response::Status allocateDataStructures() override;

   // Data members
  protected:
   HyPerDelivery::AccumulateType mAccumulateType = HyPerDelivery::CONVOLVE;

   char *mAccumulateTypeString         = nullptr;
   bool mUpdateGSynFromPostPerspective = false;

   // Whether to check if pre-layer is spiking and, if it is not,
   // scale activity by dt to convert it to a spike count
   bool mConvertRateToSpikeCount = false;

   HyPerDelivery *mDeliveryIntern = nullptr;
}; // end class HyPerDeliveryFacade

} // end namespace PV

#endif // HYPERDELIVERYFACADE_HPP_
