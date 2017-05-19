/*
 * BaseConnectionProbe.cpp
 *
 *  Created on: Oct 20, 2011
 *      Author: pschultz
 */

#include "BaseConnectionProbe.hpp"

namespace PV {

BaseConnectionProbe::BaseConnectionProbe() { initialize_base(); }

BaseConnectionProbe::BaseConnectionProbe(const char *probeName, HyPerCol *hc) {
   initialize_base();
   initialize(probeName, hc);
}

BaseConnectionProbe::~BaseConnectionProbe() {}

int BaseConnectionProbe::initialize_base() {
   targetConn = NULL;
   return PV_SUCCESS;
}

int BaseConnectionProbe::initialize(const char *probeName, HyPerCol *hc) {
   int status = BaseProbe::initialize(probeName, hc);
   return status;
}

void BaseConnectionProbe::ioParam_targetName(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamString(ioFlag, name, "targetConnection", &targetName, NULL, false);
   if (targetName == NULL) {
      BaseProbe::ioParam_targetName(ioFlag);
   }
}

int BaseConnectionProbe::communicateInitInfo(CommunicateInitInfoMessage const *message) {
   int status = BaseProbe::communicateInitInfo(message);
   if (status != PV_SUCCESS) {
      return status;
   }
   targetConn = dynamic_cast<BaseConnection *>(message->lookup(std::string(targetName)));
   if (targetConn == nullptr) {
      ErrorLog().printf(
            "%s, rank %d process: targetConnection \"%s\" is "
            "not a connection in the column.\n",
            getDescription_c(),
            getParent()->columnId(),
            targetName);
      status = PV_FAILURE;
   }
   MPI_Barrier(getParent()->getCommunicator()->communicator());
   if (status != PV_SUCCESS) {
      exit(EXIT_FAILURE);
   }
   targetConn->insertProbe(this);
   return status;
}

} // end of namespace PV
