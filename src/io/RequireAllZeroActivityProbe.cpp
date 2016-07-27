/*
 * RequireAllZeroActivityProbe.cpp
 *
 *  Created on: Mar 26, 2014
 *      Author: pschultz
 */

#include "RequireAllZeroActivityProbe.hpp"

namespace PV {

RequireAllZeroActivityProbe::RequireAllZeroActivityProbe(const char * probeName, HyPerCol * hc) {
   initialize_base();
   initRequireAllZeroActivityProbe(probeName, hc);
}

RequireAllZeroActivityProbe::RequireAllZeroActivityProbe() {
   initialize_base();
}

int RequireAllZeroActivityProbe::initialize_base() {
   return PV_SUCCESS;
}

int RequireAllZeroActivityProbe::initRequireAllZeroActivityProbe(const char * probeName, HyPerCol * hc) {
   int status = StatsProbe::initStatsProbe(probeName, hc);
   return status;
}

int RequireAllZeroActivityProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
    int status = StatsProbe::ioParamsFillGroup(ioFlag);
    ioParam_exitOnFailure(ioFlag);
    ioParam_immediateExitOnFailure(ioFlag);
    return status;
}

void RequireAllZeroActivityProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   requireType(BufActivity);
}

void RequireAllZeroActivityProbe::ioParam_exitOnFailure(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "exitOnFailure", &exitOnFailure, exitOnFailure);
}

void RequireAllZeroActivityProbe::ioParam_immediateExitOnFailure(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "exitOnFailure"));
   if (exitOnFailure) {
      parent->ioParamValue(ioFlag, name, "immediateExitOnFailure", &immediateExitOnFailure, immediateExitOnFailure);
   }
   else {
      immediateExitOnFailure = false;
   }
}

int RequireAllZeroActivityProbe::outputState(double timed) {
   int status = StatsProbe::outputState(timed);
   for(int b = 0; b < getParent()->getNBatch(); b++){
      if (nnz[b]!=0) {
         if (!nonzeroFound) {
            nonzeroTime = timed;
         }
         nonzeroFound = true;
         nonzeroFoundMessage(nonzeroTime, parent->columnId()==0, immediateExitOnFailure);
      }
   }
   return status;
}

void RequireAllZeroActivityProbe::nonzeroFoundMessage(double badTime, bool isRoot, bool fatalError) {
   if (isRoot) {
      std::stringstream message("");
      message << getDescription_c() << ": Nonzero activity found at time " << badTime << "\n";
      if (fatalError) {
         pvError() << message.str();
      }
      else {
         pvWarn() << message.str();
      }
   }
   if (fatalError) {
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
}

RequireAllZeroActivityProbe::~RequireAllZeroActivityProbe() {
   //We check for exits on failure in destructor
   if(exitOnFailure && getNonzeroFound()) {
      nonzeroFoundMessage(nonzeroTime, parent->columnId()==0, true/*fatalError*/);
   }
}

} /* namespace PV */
