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
   pvAssert(getTargetConn());
   if (getTargetHyPerConn() == nullptr) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: targetConn \"%s\" must be a HyPerConn or "
               "HyPerConn-derived class.\n",
               getDescription_c(),
               mTargetConn->getName());
      }
      MPI_Barrier(parent->getCommunicator()->globalCommunicator());
      exit(EXIT_FAILURE);
   }
   return status;
}

bool BaseHyPerConnProbe::needRecalc(double timevalue) {
   return getLastUpdateTime() < getTargetHyPerConn()->getLastUpdateTime();
}

double BaseHyPerConnProbe::referenceUpdateTime() const {
   return getTargetHyPerConn()->getLastUpdateTime();
}

BaseHyPerConnProbe::~BaseHyPerConnProbe() {}

} /* namespace PV */
