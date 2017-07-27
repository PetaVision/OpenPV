/*
 * BaseHyPerConnProbe.cpp
 *
 *  Created on: Oct 28, 2014
 *      Author: pschultz
 */

#include "BaseHyPerConnProbe.hpp"

namespace PV {

BaseHyPerConnProbe::BaseHyPerConnProbe(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

BaseHyPerConnProbe::BaseHyPerConnProbe() { initialize_base(); }

int BaseHyPerConnProbe::initialize_base() {
   targetHyPerConn = NULL;
   return PV_SUCCESS;
}

int BaseHyPerConnProbe::initialize(const char *name, HyPerCol *hc) {
   return BaseConnectionProbe::initialize(name, hc);
}

int BaseHyPerConnProbe::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   int status = BaseConnectionProbe::communicateInitInfo(message);
   assert(getTargetConn());
   targetHyPerConn = dynamic_cast<HyPerConn *>(targetConn);
   if (targetHyPerConn == NULL) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "%s: targetConn \"%s\" must be a HyPerConn or "
               "HyPerConn-derived class.\n",
               getDescription_c(),
               targetConn->getName());
      }
      status = PV_FAILURE;
   }
   return status;
}

bool BaseHyPerConnProbe::needRecalc(double timevalue) {
   return this->getLastUpdateTime() < targetHyPerConn->getLastUpdateTime();
}

double BaseHyPerConnProbe::referenceUpdateTime() const {
   return targetHyPerConn->getLastUpdateTime();
}

BaseHyPerConnProbe::~BaseHyPerConnProbe() {}

} /* namespace PV */
