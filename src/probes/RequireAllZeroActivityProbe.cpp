/*
 * RequireAllZeroActivityProbe.cpp
 *
 *  Created on: Mar 26, 2014
 *      Author: pschultz
 */

#include "RequireAllZeroActivityProbe.hpp"
#include "layers/HyPerLayer.hpp"

namespace PV {

RequireAllZeroActivityProbe::RequireAllZeroActivityProbe(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   initialize_base();
   initialize(name, params, comm);
}

RequireAllZeroActivityProbe::RequireAllZeroActivityProbe() { initialize_base(); }

int RequireAllZeroActivityProbe::initialize_base() { return PV_SUCCESS; }

void RequireAllZeroActivityProbe::initialize(
      const char *name,
      PVParams *params,
      Communicator const *comm) {
   StatsProbe::initialize(name, params, comm);
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
   parameters()->ioParamValue(ioFlag, name, "exitOnFailure", &exitOnFailure, exitOnFailure);
}

void RequireAllZeroActivityProbe::ioParam_immediateExitOnFailure(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "exitOnFailure"));
   if (exitOnFailure) {
      parameters()->ioParamValue(
            ioFlag,
            name,
            "immediateExitOnFailure",
            &immediateExitOnFailure,
            immediateExitOnFailure);
   }
   else {
      immediateExitOnFailure = false;
   }
}

Response::Status RequireAllZeroActivityProbe::outputState(double simTime, double deltaTime) {
   auto status = StatsProbe::outputState(simTime, deltaTime);
   if (!Response::completed(status)) {
      Fatal() << getDescription() << ": StatsProbe::outputState failed at time " << simTime
              << ".\n";
   }
   int const nbatch = targetLayer->getLayerLoc()->nbatch;
   for (int b = 0; b < nbatch; b++) {
      if (nnz[b] != 0) {
         if (!nonzeroFound) {
            nonzeroTime = simTime;
         }
         nonzeroFound = true;
         nonzeroFoundMessage(
               nonzeroTime, mCommunicator->globalCommRank() == 0, immediateExitOnFailure);
      }
   }
   return Response::SUCCESS;
}

void RequireAllZeroActivityProbe::nonzeroFoundMessage(
      double badTime,
      bool isRoot,
      bool fatalError) {
   if (isRoot) {
      std::stringstream message("");
      message << getDescription_c() << ": Nonzero activity found at time " << badTime << "\n";
      if (fatalError) {
         Fatal() << message.str();
      }
      else {
         WarnLog() << message.str();
      }
   }
   if (fatalError) {
      MPI_Barrier(mCommunicator->communicator());
      exit(EXIT_FAILURE);
   }
}

RequireAllZeroActivityProbe::~RequireAllZeroActivityProbe() {
   // We check for exits on failure in destructor
   if (exitOnFailure && getNonzeroFound()) {
      nonzeroFoundMessage(nonzeroTime, mCommunicator->globalCommRank() == 0, true /*fatalError*/);
   }
}

} /* namespace PV */
