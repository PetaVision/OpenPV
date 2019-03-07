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
  public:
   enum AccumulateType { UNDEFINED, CONVOLVE, STOCHASTIC };

   HyPerDelivery(char const *name, PVParams *params, Communicator const *comm);

   virtual ~HyPerDelivery();

   void setConnectionData(ConnectionData *connectionData);

   virtual void deliver(float *destBuffer) override = 0;

   virtual bool isAllInputReady() const override;

  protected:
   HyPerDelivery();

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   double convertToRateDeltaTimeFactor(double timeConstantTau, double deltaTime) const;

   // Data members
  protected:
   AccumulateType mAccumulateType      = CONVOLVE;
   bool mUpdateGSynFromPostPerspective = false;

   float mDeltaTimeFactor    = 1.0f;
   WeightsPair *mWeightsPair = nullptr;
   ArborList *mArborList     = nullptr;

}; // end class HyPerDelivery

} // end namespace PV

#endif // HYPERDELIVERY_HPP_
