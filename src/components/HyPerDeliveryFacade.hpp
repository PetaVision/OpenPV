/*
 * HyPerDeliveryFacade.hpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#ifndef HYPERDELIVERYFACADE_HPP_
#define HYPERDELIVERYFACADE_HPP_

#include "components/BaseDelivery.hpp"
#include "components/HyPerDelivery.hpp"

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
    * If true, the connection loops over presynaptic neurons, and each pre-neuron pushes to its
    * region of influence. This allows efficiency for sparse pre-layers, but requires extra memory
    * to manage potential collisions as multiple pre-neurons write to the same post-neuron.
    */
   virtual void ioParam_updateGSynFromPostPerspective(enum ParamsIOFlag ioFlag);
   /** @} */ // End of list of HyPerDeliveryFacade parameters.

  public:
   HyPerDeliveryFacade(char const *name, HyPerCol *hc);

   virtual ~HyPerDeliveryFacade();

   virtual void deliver(Weights *weights) override;

   virtual void deliverUnitInput(Weights *weights, float *recvBuffer) override;

   HyPerDelivery::AccumulateType getAccumulateType() const { return mAccumulateType; }

   bool getUpdateGSynFromPostPerspective() const { return mUpdateGSynFromPostPerspective; }

   bool getConvertRateToSpikeCount() const { return mConvertRateToSpikeCount; }

  protected:
   HyPerDeliveryFacade();

   int initialize(char const *name, HyPerCol *hc);

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   void createDeliveryIntern();

   virtual int allocateDataStructures() override;

   double convertToRateDeltaTimeFactor(double timeConstantTau) const;

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