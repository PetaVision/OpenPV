/*
 * BaseHyPerConnProbe.cpp
 *
 *  Created on: Oct 28, 2014
 *      Author: pschultz
 */

#include "BaseHyPerConnProbe.hpp"
#include "components/WeightsPair.hpp"

namespace PV {

BaseHyPerConnProbe::BaseHyPerConnProbe(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

BaseHyPerConnProbe::BaseHyPerConnProbe() {}

void BaseHyPerConnProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   BaseConnectionProbe::initialize(name, params, comm);
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
      if (mCommunicator->globalCommRank() == 0) {
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

double BaseHyPerConnProbe::referenceUpdateTime(double simTime) const {
   return mWeights->getTimestamp();
}

BaseHyPerConnProbe::~BaseHyPerConnProbe() {}

} /* namespace PV */
