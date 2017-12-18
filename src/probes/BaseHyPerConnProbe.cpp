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

int BaseHyPerConnProbe::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status = BaseConnectionProbe::communicateInitInfo(message);
   assert(getTargetConn());
   if (getTargetHyPerConn() == nullptr) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: targetConn \"%s\" must be a HyPerConn or "
               "HyPerConn-derived class.\n",
               getDescription_c(),
               mTargetConn->getName());
      }
      status = PV_FAILURE;
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
