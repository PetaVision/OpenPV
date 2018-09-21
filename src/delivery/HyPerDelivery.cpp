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

void HyPerDelivery::setObjectType() { mObjectType = "HyPerDelivery"; }

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
   parameters()->ioParamValue(
         ioFlag,
         this->getName(),
         "convertRateToSpikeCount",
         &mConvertRateToSpikeCount,
         mConvertRateToSpikeCount /*default value*/);
}

Response::Status
HyPerDelivery::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = BaseDelivery::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   mWeightsPair = mapLookupByType<WeightsPair>(message->mHierarchy);
   pvAssert(mWeightsPair);
   if (!mWeightsPair->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }

   mArborList = mapLookupByType<ArborList>(message->mHierarchy);
   pvAssert(mArborList);
   if (!mArborList->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   return Response::SUCCESS;
}

Response::Status HyPerDelivery::allocateDataStructures() {
   auto status      = BaseDelivery::allocateDataStructures();
   mDeltaTimeFactor = 1.0f; // Default value
   if (!Response::completed(status)) {
      return status;
   }
   if (mAccumulateType == STOCHASTIC) {
      mDeltaTimeFactor = (float)parent->getDeltaTime();
   }
   else if (mConvertRateToSpikeCount and !mPreLayer->activityIsSpiking()) {
      auto layerInputBuffer = mPostLayer->getComponentByType<LayerInputBuffer>();
      if (layerInputBuffer) {
         if (!layerInputBuffer->getDataStructuresAllocatedFlag()) {
            return status + Response::POSTPONE;
         }
         double timeConstant = layerInputBuffer->getChannelTimeConstant(mChannelCode);
         mDeltaTimeFactor    = (float)convertToRateDeltaTimeFactor(timeConstant);
      }
   }
   return Response::SUCCESS;
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
      int const numArbors = mArborList->getNumAxonalArbors();
      for (int a = 0; a < numArbors; a++) {
         isReady &= getPreLayer()->isExchangeFinished(mArborList->getDelay(a));
      }
   }
   return isReady;
}

} // end namespace PV
