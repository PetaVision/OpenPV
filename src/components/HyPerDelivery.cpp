/*
 * HyPerDelivery.cpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#include "HyPerDelivery.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

HyPerDelivery::HyPerDelivery(char const *name, HyPerCol *hc) { initialize(name, hc); }

HyPerDelivery::HyPerDelivery() {}

HyPerDelivery::~HyPerDelivery() {}

int HyPerDelivery::initialize(char const *name, HyPerCol *hc) {
   return BaseDelivery::initialize(name, hc);
}

int HyPerDelivery::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = PV_SUCCESS;
   // Only read params because HyPerDeliveryFacade will read/write them too.
   // The facade needs to read the params in order to determine which HyPerDelivery subclass
   // to instantiate.
   if (ioFlag == PARAMS_IO_READ) {
      status = BaseDelivery::ioParamsFillGroup(ioFlag);
   }
   ioParam_convertRateToSpikeCount(ioFlag);
   return PV_SUCCESS;
}

void HyPerDelivery::ioParam_convertRateToSpikeCount(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   parent->parameters()->ioParamValue(
         ioFlag,
         this->getName(),
         "convertRateToSpikeCount",
         &mConvertRateToSpikeCount,
         mConvertRateToSpikeCount /*default value*/);
}

int HyPerDelivery::allocateDataStructures() {
   int status = BaseDelivery::allocateDataStructures();
   float dtFactor;
   if (mAccumulateType == STOCHASTIC or mPreLayer->activityIsSpiking()) {
      mDeltaTimeFactor = (float)parent->getDeltaTime();
   }
   else if (mAccumulateType == CONVOLVE) {
      mDeltaTimeFactor =
            (float)convertToRateDeltaTimeFactor(mPostLayer->getChannelTimeConst(mChannelCode));
   }
   return status;
}

double HyPerDelivery::convertToRateDeltaTimeFactor(double timeConstantTau) const {
   double dtFactor = 1.0;
   if (mConvertRateToSpikeCount) {
      double dt = parent->getDeltaTime();
      if (timeConstantTau > 0) {
         double exp_dt_tau = exp(-dt / timeConstantTau);
         dtFactor          = (1.0 - exp_dt_tau) / exp_dt_tau;
         // the above factor was chosen so that for a constant input of G_SYN to an excitatory
         // conductance G_EXC, then G_EXC -> G_SYN as t -> inf
      }
      else {
         dtFactor = dt;
      }
   }
   return dtFactor;
}

} // end namespace PV
