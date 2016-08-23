/*
 * AdaptiveTimestepProbe.cpp
 *
 *  Created on: Aug 18, 2016
 *      Author: pschultz
 */

#include <io/AdaptiveTimestepProbe.hpp>
#include "columns/Messages.hpp"

namespace PV {

AdaptiveTimestepProbe::AdaptiveTimestepProbe(char const * name, HyPerCol * hc) {
   initialize(name, hc);
}

AdaptiveTimestepProbe::AdaptiveTimestepProbe() {
}

AdaptiveTimestepProbe::~AdaptiveTimestepProbe() {
   delete mAdaptiveTimestepController;
}

int AdaptiveTimestepProbe::initialize(char const * name, HyPerCol * hc) {
   int status = ColProbe::initialize(name, hc);
   return status;
}

int AdaptiveTimestepProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ColProbe::ioParamsFillGroup(ioFlag);
   ioParam_dtScaleMax(ioFlag);
   ioParam_dtScaleMin(ioFlag);
   ioParam_dtChangeMax(ioFlag);
   ioParam_dtChangeMin(ioFlag);
   ioParam_dtMinToleratedTimeScale(ioFlag);
   ioParam_writeTimescales(ioFlag);
   ioParam_writeTimeScaleFieldnames(ioFlag);
   return status;
}

void AdaptiveTimestepProbe::ioParam_targetName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "targetName", &targetName);
}

void AdaptiveTimestepProbe::ioParam_dtScaleMax(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "dtScaleMax", &mTimeScaleMaxBase, mTimeScaleMaxBase);
}

void AdaptiveTimestepProbe::ioParam_dtScaleMin(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "dtScaleMin", &mTimeScaleMin, mTimeScaleMin);
}

void AdaptiveTimestepProbe::ioParam_dtMinToleratedTimeScale(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "dtMinToleratedTimeScale", &mDtMinToleratedTimeScale, mDtMinToleratedTimeScale);
}

void AdaptiveTimestepProbe::ioParam_dtChangeMax(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "dtChangeMax", &mChangeTimeScaleMax, mChangeTimeScaleMax);
}

void AdaptiveTimestepProbe::ioParam_dtChangeMin(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "dtChangeMin", &mChangeTimeScaleMin, mChangeTimeScaleMin);
}

void AdaptiveTimestepProbe::ioParam_writeTimescales(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "writeTimescales", &mWriteTimescales, mWriteTimescales);
}

void AdaptiveTimestepProbe::ioParam_writeTimeScaleFieldnames(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "writeTimescales"));
   if (mWriteTimescales) {
     parent->ioParamValue(ioFlag, name, "writeTimeScaleFieldnames", &mWriteTimeScaleFieldnames, mWriteTimeScaleFieldnames);
   }
}

int AdaptiveTimestepProbe::communicateInitInfo() {
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

int AdaptiveTimestepProbe::allocateDataStructures() {
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
   mAdaptiveTimestepController = new AdaptiveTimestepController(
         getName(),
         getNumValues(),
         parent->getDeltaTime(),
         mTimeScaleMaxBase,
         mTimeScaleMin,
         mDtMinToleratedTimeScale,
         mChangeTimeScaleMax,
         mChangeTimeScaleMin,
         mWriteTimescales,
         mWriteTimeScaleFieldnames,
         parent->getCommunicator(),
         parent->getVerifyWrites()
   );
   return status;
}

int AdaptiveTimestepProbe::checkpointRead(const char * cpDir, double * timeptr) {
   int status = ColProbe::checkpointRead(cpDir, timeptr);
   if (status==PV_SUCCESS) {
      status = mAdaptiveTimestepController->checkpointRead(cpDir, timeptr);
   }
   return status;
}

int AdaptiveTimestepProbe::checkpointWrite(const char * cpDir) {
   int status = ColProbe::checkpointWrite(cpDir);
   if (status==PV_SUCCESS) {
      status = mAdaptiveTimestepController->checkpointWrite(cpDir);
   }
   return status;
}

int AdaptiveTimestepProbe::respond(std::shared_ptr<BaseMessage> message) {
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

int AdaptiveTimestepProbe::respondAdaptTimestep(AdaptTimestepMessage const * message) {
   return getValues(parent->simulationTime());
}

// AdaptiveTimestepProbe::calcValues calls targetProbe->getValues() and passes the
// result to mAdaptiveTimestepController->calcTimesteps() to use as timeScaleTrue.
// mAdaptiveTimestepController->calcTimesteps() returns timeScale, checks it against
// dtMinToleratedTimeScale (unless it doesn't) and copies the result into probeValues.
// AdaptiveTimestepProbe should also process the triggering and only involve
// mAdaptiveTimestepController when triggering doesn't happen.

int AdaptiveTimestepProbe::calcValues(double timeValue) {
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
   for (int b=0; b<getNumValues(); b++) {
      double rawTimescale = rawProbeValues.at(b);
      if (rawTimescale > 0 && rawTimescale < mDtMinToleratedTimeScale) {
         if (parent->getCommunicator()->globalCommRank()) {
            if (getNumValues()==1) {
               pvErrorNoExit() << getDescription() << ": target probe \"" <<
                     ", has time scale " << rawTimescale <<
                     ", less than dtMinToleratedTimeScale=" <<
                     mDtMinToleratedTimeScale << ".\n";
            }
            else {
               pvErrorNoExit() << getDescription() << ": target probe \"" <<
                     mTargetProbe->getName() << "\"" <<
                     ", batch element " << b <<
                     ", has time scale " << rawTimescale <<
                     ", less than dtMinToleratedTimeScale=" <<
                     mDtMinToleratedTimeScale << ".\n";
            }
         }
         MPI_Barrier(parent->getCommunicator()->globalCommunicator());
         exit(EXIT_FAILURE);
      }
   }
   std::vector<double> const& timeSteps = mAdaptiveTimestepController->calcTimesteps(timeValue, rawProbeValues);
   memcpy(getValuesBuffer(), timeSteps.data(), sizeof(double)*getNumValues());
   return PV_SUCCESS;
}

int AdaptiveTimestepProbe::outputState(double timeValue) {
   if (outputStream) { mAdaptiveTimestepController->writeTimestepInfo(timeValue, output()); }
   return PV_SUCCESS;
}

} /* namespace PV */
