/*
 * HyPerDelivery.cpp
 *
 *  Created on: Aug 24, 2017
 *      Author: Pete Schultz
 */

#include "HyPerDelivery.hpp"

namespace PV {

HyPerDelivery::HyPerDelivery(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

HyPerDelivery::HyPerDelivery() {}

HyPerDelivery::~HyPerDelivery() {}

void HyPerDelivery::initialize(char const *name, PVParams *params, Communicator const *comm) {
   BaseDelivery::initialize(name, params, comm);
}

void HyPerDelivery::setObjectType() { mObjectType = "HyPerDelivery"; }

void HyPerDelivery::ioParam_receiveGpu(enum ParamsIOFlag ioFlag) {
   // Don't call handleUnnecessaryParameter here because that will generate a warning.
   // HyPerDeliver-derived classes don't need this parameter, but in the usual situation,
   // the parameter is read by HyPerDeliverCreator, which does need the parameter.
   // Hence a warning generated here would be misleading.
   if (ioFlag == PARAMS_IO_READ) {
      bool receiveGpu = parameters()->value(
            name, "receiveGpu", mCorrectReceiveGpu, false /*don't warn if absent*/);
      FatalIf(
            receiveGpu != mCorrectReceiveGpu,
            "%s has receiveGpu set to %s in params, but requires %s to be %s.\n",
            getDescription_c(),
            receiveGpu ? "true" : "false",
            mCorrectReceiveGpu ? "true" : "false");
   }
}

Response::Status
HyPerDelivery::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = BaseDelivery::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   mWeightsPair = message->mObjectTable->findObject<WeightsPair>(getName());
   pvAssert(mWeightsPair);
   if (!mWeightsPair->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }

   mArborList = message->mObjectTable->findObject<ArborList>(getName());
   pvAssert(mArborList);
   if (!mArborList->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   return Response::SUCCESS;
}

double HyPerDelivery::convertToRateDeltaTimeFactor(double timeConstantTau, double deltaTime) const {
   return std::exp(deltaTime / timeConstantTau) - 1.0;
   // the above factor was chosen so that for a constant input of G_SYN to an excitatory
   // conductance G_EXC, then G_EXC -> G_SYN as t -> inf
}

bool HyPerDelivery::isAllInputReady() const {
   bool isReady = true;
   if (getChannelCode() != CHANNEL_NOUPDATE) {
      int const numArbors = mArborList->getNumAxonalArbors();
      for (int a = 0; a < numArbors; a++) {
         isReady &= mPreData->isExchangeFinished(mArborList->getDelay(a));
      }
   }
   return isReady;
}

} // end namespace PV
