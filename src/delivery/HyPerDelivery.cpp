/*
 * HyPerDelivery.cpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#include "HyPerDelivery.hpp"
#include "columns/HyPerCol.hpp"
#include "utils/MapLookupByType.hpp"

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
   parent->parameters()->ioParamValue(
         ioFlag,
         this->getName(),
         "convertRateToSpikeCount",
         &mConvertRateToSpikeCount,
         mConvertRateToSpikeCount /*default value*/);
}

int HyPerDelivery::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status   = BaseDelivery::communicateInitInfo(message);
   mWeightsPair = mapLookupByType<WeightsPair>(message->mHierarchy, getDescription());
   pvAssert(mWeightsPair);
   return status;
}

int HyPerDelivery::allocateDataStructures() {
   int status = BaseDelivery::allocateDataStructures();
   if (mAccumulateType == STOCHASTIC) {
      mDeltaTimeFactor = (float)parent->getDeltaTime();
   }
   else if (mConvertRateToSpikeCount and !mPreLayer->activityIsSpiking()) {
      mDeltaTimeFactor =
            (float)convertToRateDeltaTimeFactor(mPostLayer->getChannelTimeConst(mChannelCode));
   }
   else {
      mDeltaTimeFactor = 1.0f;
   }
   return status;
}

double HyPerDelivery::convertToRateDeltaTimeFactor(double timeConstantTau) const {
   double dt = parent->getDeltaTime();
   double dtFactor;
   if (timeConstantTau > 0) {
      dtFactor = std::exp(dt / timeConstantTau) - 1.0;
      // the above factor was chosen so that for a constant input of G_SYN to an excitatory
      // conductance G_EXC, then G_EXC -> G_SYN as t -> inf
   }
   else {
      dtFactor = dt;
   }
   return dtFactor;
}

bool HyPerDelivery::isAllInputReady() {
   bool isReady = true;
   if (getChannelCode() != CHANNEL_NOUPDATE) {
      int const numArbors = mConnectionData->getNumAxonalArbors();
      for (int a = 0; a < numArbors; a++) {
         isReady &= getPreLayer()->isExchangeFinished(mConnectionData->getDelay(a));
      }
   }
   return isReady;
}

} // end namespace PV
