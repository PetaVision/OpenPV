/*
 * AdaptiveTimeScaleProbe.cpp
 *
 *  Created on: Aug 18, 2016
 *      Author: pschultz
 */

#include <io/AdaptiveTimeScaleProbe.hpp>
#include "columns/Messages.hpp"

namespace PV {

AdaptiveTimeScaleProbe::AdaptiveTimeScaleProbe(char const * name, HyPerCol * hc) {
   initialize(name, hc);
}

AdaptiveTimeScaleProbe::AdaptiveTimeScaleProbe() {
}

AdaptiveTimeScaleProbe::~AdaptiveTimeScaleProbe() {
   delete mAdaptiveTimeScaleController;
}

int AdaptiveTimeScaleProbe::initialize(char const * name, HyPerCol * hc) {
   int status = ColProbe::initialize(name, hc);
   return status;
}

int AdaptiveTimeScaleProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ColProbe::ioParamsFillGroup(ioFlag);
   ioParam_baseMax(ioFlag);
   ioParam_baseMin(ioFlag);
   ioParam_tauFactor(ioFlag);
   ioParam_growthFactor(ioFlag);
   ioParam_dtMinToleratedTimeScale(ioFlag);
   ioParam_writeTimeScales(ioFlag);
   ioParam_writeTimeScaleFieldnames(ioFlag);
   return status;
}

void AdaptiveTimeScaleProbe::ioParam_targetName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "targetName", &targetName);
}

void AdaptiveTimeScaleProbe::ioParam_baseMax(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "baseMax", &mBaseMax, mBaseMax);
}

void AdaptiveTimeScaleProbe::ioParam_baseMin(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "baseMin", &mBaseMin, mBaseMin);
}

void AdaptiveTimeScaleProbe::ioParam_dtMinToleratedTimeScale(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ && parent->parameters()->present(getName(), "dtScaleMin")) {
      if (parent->getCommunicator()->commRank()==0) {
         pvErrorNoExit() << "The dtMinToleratedTimeScale parameter has been removed.\n";
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
}

void AdaptiveTimeScaleProbe::ioParam_tauFactor(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "tauFactor", &tauFactor, tauFactor);
}

void AdaptiveTimeScaleProbe::ioParam_growthFactor(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "growthFactor", &mGrowthFactor, mGrowthFactor);
}

void AdaptiveTimeScaleProbe::ioParam_writeTimeScales(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "writeTimeScales", &mWriteTimeScales, mWriteTimeScales);
}

void AdaptiveTimeScaleProbe::ioParam_writeTimeScaleFieldnames(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "writeTimeScales"));
   if (mWriteTimeScales) {
     parent->ioParamValue(ioFlag, name, "writeTimeScaleFieldnames", &mWriteTimeScaleFieldnames, mWriteTimeScaleFieldnames);
   }
}

int AdaptiveTimeScaleProbe::communicateInitInfo() {
   int status = ColProbe::communicateInitInfo();
   mTargetProbe = parent->getBaseProbeFromName(targetName);
   if (mTargetProbe==nullptr) {
      if (parent->getCommunicator()->commRank()==0) {
         pvError() << getDescription() << ": targetName \"" << targetName << "\" is not a probe in the HyPerCol.\n";
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   return status;
}

int AdaptiveTimeScaleProbe::allocateDataStructures() {
   int status = ColProbe::allocateDataStructures();
   if (mTargetProbe->getNumValues() != getNumValues()) {
      if (parent->getCommunicator()->commRank()==0) {
         pvError() << getDescription() << ": target probe \"" <<
               mTargetProbe->getDescription() <<
               "\" does not have the correct numValues (" <<
               mTargetProbe->getNumValues() << " instead of " <<
               getNumValues() << ").\n";
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   mAdaptiveTimeScaleController = new AdaptiveTimeScaleController(
         getName(),
         getNumValues(),
         mBaseMax,
         mBaseMin,
         tauFactor,
         mGrowthFactor,
         mWriteTimeScales,
         mWriteTimeScaleFieldnames,
         parent->getCommunicator(),
         parent->getVerifyWrites()
   );
   return status;
}

int AdaptiveTimeScaleProbe::checkpointRead(const char * cpDir, double * timeptr) {
   int status = ColProbe::checkpointRead(cpDir, timeptr);
   if (status==PV_SUCCESS) {
      status = mAdaptiveTimeScaleController->checkpointRead(cpDir, timeptr);
   }
   return status;
}

int AdaptiveTimeScaleProbe::checkpointWrite(const char * cpDir) {
   int status = ColProbe::checkpointWrite(cpDir);
   if (status==PV_SUCCESS) {
      status = mAdaptiveTimeScaleController->checkpointWrite(cpDir);
   }
   return status;
}

int AdaptiveTimeScaleProbe::respond(std::shared_ptr<BaseMessage> message) {
   int status = ColProbe::respond(message);
   if (message==nullptr) {
      return status;
   }
   else if (AdaptTimestepMessage const * castMessage = dynamic_cast<AdaptTimestepMessage const*>(message.get())) {
      return respondAdaptTimestep(castMessage);
   }
   else {
      return status;
   }
}

int AdaptiveTimeScaleProbe::respondAdaptTimestep(AdaptTimestepMessage const * message) {
   return getValues(parent->simulationTime());
}

// AdaptiveTimeScaleProbe::calcValues calls targetProbe->getValues() and passes the
// result to mAdaptiveTimeScaleController->calcTimesteps() to use as timeScaleTrue.
// mAdaptiveTimeScaleController->calcTimesteps() returns timeScale, checks it against
// dtMinToleratedTimeScale (unless it doesn't) and copies the result into probeValues.
// AdaptiveTimeScaleProbe should also process the triggering and only involve
// mAdaptiveTimeScaleController when triggering doesn't happen.

int AdaptiveTimeScaleProbe::calcValues(double timeValue) {
   std::vector<double> rawProbeValues;
   bool triggersNow = false;
   if (triggerLayer) {
      double triggerTime = triggerLayer->getNextUpdateTime() - triggerOffset;
      triggersNow = fabs(timeValue - triggerTime) < (parent->getDeltaTime()/2);
   }
   if (triggersNow) {
      rawProbeValues.assign(getNumValues(), -1.0);
   }
   else {
      mTargetProbe->getValues(timeValue, &rawProbeValues);
   }
   pvAssert(rawProbeValues.size()==getNumValues()); // In allocateDataStructures, we checked that mTargetProbe has a compatible size.
   std::vector<double> const& timeSteps = mAdaptiveTimeScaleController->calcTimesteps(timeValue, rawProbeValues);
   memcpy(getValuesBuffer(), timeSteps.data(), sizeof(double)*getNumValues());
   return PV_SUCCESS;
}

int AdaptiveTimeScaleProbe::outputState(double timeValue) {
   if (outputStream) { mAdaptiveTimeScaleController->writeTimestepInfo(timeValue, output()); }
   return PV_SUCCESS;
}

} /* namespace PV */
