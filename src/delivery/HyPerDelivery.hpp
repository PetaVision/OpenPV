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
 * A pure virtual class for delivery components for HyPerConns. It is typically created by a
 * HyPerDeliveryCreator, which reads the receiveGpu parameter to decide which HyPerDelivery
 * subclass to create; hence the receiveGpu parameter in this subclass does nothing.
 */
class HyPerDelivery : public BaseDelivery {
  protected:
   /**
    * The HyPerDeliveryCreator object reads this parameter and creates the appropriate
    * HyPerDelivery-derived object based on its value; hence the derived class knows
    * the value of receiveGpu just from having been instantiated.
    */
   void ioParam_receiveGpu(enum ParamsIOFlag ioFlag) override;

  public:
   HyPerDelivery(char const *name, PVParams *params, Communicator const *comm);

   virtual ~HyPerDelivery();

   virtual void deliver(float *destBuffer) override = 0;

   virtual bool isAllInputReady() const override;

  protected:
   HyPerDelivery();

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   double convertToRateDeltaTimeFactor(double timeConstantTau, double deltaTime) const;

   // Data members
  protected:
   float mDeltaTimeFactor    = 1.0f;
   WeightsPair *mWeightsPair = nullptr;
   ArborList *mArborList     = nullptr;

}; // end class HyPerDelivery

} // end namespace PV

#endif // HYPERDELIVERY_HPP_
