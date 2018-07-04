/*
 * BaseHyPerConnProbe.cpp
 *
 *  Created on: Oct 28, 2014
 *      Author: pschultz
 */

#include "BaseHyPerConnProbe.hpp"

namespace PV {

BaseHyPerConnProbe::BaseHyPerConnProbe(const char *name, HyPerCol *hc) { initialize(name, hc); }

BaseHyPerConnProbe::BaseHyPerConnProbe() {}

int BaseHyPerConnProbe::initialize(const char *name, HyPerCol *hc) {
   return BaseConnectionProbe::initialize(name, hc);
}

Response::Status
BaseHyPerConnProbe::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = BaseConnectionProbe::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   pvAssert(mTargetConn);

   auto *weightsPair = mTargetConn->getComponentByType<WeightsPair>();
   FatalIf(
         weightsPair == nullptr,
         "%s target connection \"%s\" does not have a WeightsPair component.\n",
         getDescription_c(),
         mTargetConn->getName());
   if (!weightsPair->getInitInfoCommunicatedFlag()) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         InfoLog().printf(
               "%s must wait until target connection \"%s\" has finished its CommunicateInitInfo "
               "stage.\n",
               getDescription_c(),
               mTargetConn->getName());
      }
      return Response::POSTPONE;
   }
   weightsPair->needPre();

   mWeights = weightsPair->getPreWeights();
   pvAssert(mWeights); // Created by needPre() call.
   return Response::SUCCESS;
}

bool BaseHyPerConnProbe::needRecalc(double timevalue) {
   return getLastUpdateTime() < mWeights->getTimestamp();
}

double BaseHyPerConnProbe::referenceUpdateTime() const { return mWeights->getTimestamp(); }

BaseHyPerConnProbe::~BaseHyPerConnProbe() {}

} /* namespace PV */
