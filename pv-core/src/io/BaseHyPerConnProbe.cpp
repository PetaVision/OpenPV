/*
 * BaseHyPerConnProbe.cpp
 *
 *  Created on: Oct 28, 2014
 *      Author: pschultz
 */

#include "BaseHyPerConnProbe.hpp"

namespace PV {

BaseHyPerConnProbe::BaseHyPerConnProbe(const char * probeName, HyPerCol * hc) {
   initialize_base();
   initialize(probeName, hc);
}

BaseHyPerConnProbe::BaseHyPerConnProbe() {
   initialize_base();
}

int BaseHyPerConnProbe::initialize_base() {
   targetHyPerConn = NULL;
   return PV_SUCCESS;
}

int BaseHyPerConnProbe::initialize(const char * probeName, HyPerCol * hc) {
   return BaseConnectionProbe::initialize(probeName, hc);
}

int BaseHyPerConnProbe::communicateInitInfo() {
   int status = BaseConnectionProbe::communicateInitInfo();
   assert(getTargetConn());
   targetHyPerConn = dynamic_cast<HyPerConn *>(targetConn);
   if (targetHyPerConn==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: targetConn \"%s\" must be a HyPerConn or HyPerConn-derived class.\n",
               this->getKeyword(), this->getName(), targetConn->getName());
      }
      status = PV_FAILURE;
   }
   return status;
}

bool BaseHyPerConnProbe::needRecalc(double timevalue) {
   return this->getLastUpdateTime() < targetHyPerConn->getLastUpdateTime();
}

BaseHyPerConnProbe::~BaseHyPerConnProbe() {
}

} /* namespace PV */
