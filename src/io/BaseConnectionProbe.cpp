/*
 * BaseConnectionProbe.cpp
 *
 *  Created on: Oct 20, 2011
 *      Author: pschultz
 */

#include "BaseConnectionProbe.hpp"

namespace PV {

BaseConnectionProbe::BaseConnectionProbe() {
   initialize_base();
}

BaseConnectionProbe::BaseConnectionProbe(const char * probeName, HyPerCol * hc)
{
   initialize_base();
   initialize(probeName, hc);
}

BaseConnectionProbe::~BaseConnectionProbe() {
}

int BaseConnectionProbe::initialize_base() {
   targetConn = NULL;
   return PV_SUCCESS;
}

int BaseConnectionProbe::initialize(const char * probeName, HyPerCol * hc) {
   int status = BaseProbe::initialize(probeName, hc);
   return status;
}

void BaseConnectionProbe::ioParam_targetName(enum ParamsIOFlag ioFlag) {
   parent->ioParamString(ioFlag, name, "targetConnection", &targetName, NULL, false);
   if(targetName == NULL){
      BaseProbe::ioParam_targetName(ioFlag);
   }
}

int BaseConnectionProbe::communicateInitInfo() {
   BaseProbe::communicateInitInfo();
   int status = setTargetConn(targetName);
   if (status == PV_SUCCESS){
      targetConn->insertProbe(this);
   }
   return status;
}

int BaseConnectionProbe::setTargetConn(const char * connName) {
   int status = PV_SUCCESS;
   targetConn = getParent()->getConnFromName(connName);
   if (targetConn==NULL) {
      pvErrorNoExit().printf("%s, rank %d process: targetConnection \"%s\" is not a connection in the HyPerCol.\n",
            getDescription_c(), getParent()->columnId(), connName);
      status = PV_FAILURE;
   }
   MPI_Barrier(getParent()->icCommunicator()->communicator());
   if (status != PV_SUCCESS) {
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}


}  // end of namespace PV


