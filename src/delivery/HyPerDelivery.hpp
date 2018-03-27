/*
 * HyPerDelivery.hpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#ifndef HYPERDELIVERY_HPP_
#define HYPERDELIVERY_HPP_

#include "BaseDelivery.hpp"
#include "components/ArborList.hpp"
#include "components/WeightsPair.hpp"

namespace PV {

/**
 * The base delivery component for HyPerConns. It is owned by a HyPerDeliveryFacade object,
 * which handles the interactions between a HyPerDelivery object and the HyPerConn object.
 * The subclasses handle the individual delivery methods, based on the values of receiveGpu,
 * pvpatchAccumulateType, and updateGSynFromPostPerspective.
 */
class HyPerDelivery : public BaseDelivery {
  protected:
   /**
    * List of parameters needed from the HyPerDeliveryFacade class
    * @name HyPerDeliveryFacade Parameters
    * @{
    */

   /**
    * @brief convertRateToSpikeCount: If true, presynaptic activity is converted
    * from a rate (spikes per second) to a count (number of spikes in the timestep).
    * @details If this flag is true and the presynaptic layer is not spiking,
    * the activity will be interpreted as a spike rate, and will be converted to a
    * spike count when delivering activity to the postsynaptic GSyn buffer.
    * If this flag is false, activity will not be converted.
    */
   virtual void ioParam_convertRateToSpikeCount(enum ParamsIOFlag ioFlag);
   /** @} */ // End of list of HyPerDeliveryFacade parameters.

  public:
   enum AccumulateType { UNDEFINED, CONVOLVE, STOCHASTIC };

   HyPerDelivery(char const *name, HyPerCol *hc);

   virtual ~HyPerDelivery();

   void setConnectionData(ConnectionData *connectionData);

   virtual void deliver() override = 0;

   virtual bool isAllInputReady() override;

  protected:
   HyPerDelivery();

   int initialize(char const *name, HyPerCol *hc);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status allocateDataStructures() override;

   double convertToRateDeltaTimeFactor(double timeConstantTau) const;

   // Data members
  protected:
   AccumulateType mAccumulateType      = CONVOLVE;
   bool mUpdateGSynFromPostPerspective = false;
   bool mConvertRateToSpikeCount       = false;

   float mDeltaTimeFactor    = 1.0f;
   WeightsPair *mWeightsPair = nullptr;
   ArborList *mArborList     = nullptr;

}; // end class HyPerDelivery

} // end namespace PV

#endif // HYPERDELIVERY_HPP_
