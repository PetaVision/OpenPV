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
   parameters()->ioParamValue(ioFlag, name, "exitOnFailure", &mExitOnFailure, mExitOnFailure);
}

void RequireAllZeroActivityProbe::ioParam_immediateExitOnFailure(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "exitOnFailure"));
   if (mExitOnFailure) {
      parameters()->ioParamValue(
            ioFlag,
            name,
            "immediateExitOnFailure",
            &mImmediateExitOnFailure,
            mImmediateExitOnFailure);
   }
   else {
      mImmediateExitOnFailure = false;
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
         if (!mNonzeroFound) {
            mNonzeroTime = simTime;
         }
         mNonzeroFound = true;
         errorMessage(
               mNonzeroTime,
               "Nonzero activity found",
               mImmediateExitOnFailure);
      }
   }
   return Response::SUCCESS;
}

void RequireAllZeroActivityProbe::errorMessage(
      double badTime,
      std::string const &baseMessage,
      bool fatalError) {
   std::stringstream message("");
   message << getDescription_c() << baseMessage << " at time " << badTime << "\n";
   int nbatch = getTargetLayer()->getLayerLoc()->nbatch;
   for (int b = 0; b < nbatch; ++b) {
      float maxabs = std::max(fMax[b], -fMin[b]);
      if (maxabs > nnzThreshold) {
         message << "    batch element " << b << " has " << nnz[b]
                 << " values exceeding threshold of " << nnzThreshold << ". Max = " << fMin[b]
                 << "; Max = " << fMax[b] << "\n";
      }
   }
   ErrorLog() << message.str();
   if (fatalError) {
      MPI_Barrier(mCommunicator->communicator());
      exit(EXIT_FAILURE);
   }
}

Response::Status RequireAllZeroActivityProbe::cleanup() {
   if (mNonzeroFound) {
      errorMessage(
            mNonzeroTime,
            "Nonzero activity beginning",
            mExitOnFailure);
   }
   return Response::SUCCESS;
}

RequireAllZeroActivityProbe::~RequireAllZeroActivityProbe() {}

} /* namespace PV */
