/*
 * AdaptiveTimeScaleProbe.cpp
 *
 *  Created on: Aug 18, 2016
 *      Author: pschultz
 */

#include "AdaptiveTimeScaleProbe.hpp"
#include "columns/Messages.hpp"

namespace PV {

AdaptiveTimeScaleProbe::AdaptiveTimeScaleProbe(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   initialize(name, params, comm);
}

AdaptiveTimeScaleProbe::AdaptiveTimeScaleProbe() {}

AdaptiveTimeScaleProbe::~AdaptiveTimeScaleProbe() { delete mAdaptiveTimeScaleController; }

void AdaptiveTimeScaleProbe::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   ColProbe::initialize(name, params, comm);
}

void AdaptiveTimeScaleProbe::initMessageActionMap() {
   ColProbe::initMessageActionMap();
   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<AdaptTimestepMessage const>(msgptr);
      return respondAdaptTimestep(castMessage);
   };
   mMessageActionMap.emplace("AdaptTimestep", action);
}

int AdaptiveTimeScaleProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ColProbe::ioParamsFillGroup(ioFlag);
   ioParam_baseMax(ioFlag);
   ioParam_baseMin(ioFlag);
   ioParam_tauFactor(ioFlag);
   ioParam_growthFactor(ioFlag);
   ioParam_writeTimeScales(ioFlag);
   ioParam_writeTimeScaleFieldnames(ioFlag);
   return status;
}

void AdaptiveTimeScaleProbe::ioParam_targetName(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamStringRequired(ioFlag, name, "targetName", &targetName);
}

void AdaptiveTimeScaleProbe::ioParam_baseMax(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "baseMax", &mBaseMax, mBaseMax);
}

void AdaptiveTimeScaleProbe::ioParam_baseMin(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "baseMin", &mBaseMin, mBaseMin);
}

void AdaptiveTimeScaleProbe::ioParam_tauFactor(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "tauFactor", &tauFactor, tauFactor);
}

void AdaptiveTimeScaleProbe::ioParam_growthFactor(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "growthFactor", &mGrowthFactor, mGrowthFactor);
}

// writeTimeScales was marked obsolete Jul 27, 2017. Use textOutputFlag instead.
void AdaptiveTimeScaleProbe::ioParam_writeTimeScales(enum ParamsIOFlag ioFlag) {
   if (ioFlag != PARAMS_IO_READ) {
      return;
   }
   pvAssert(!parameters()->presentAndNotBeenRead(name, "textOutputFlag"));
   if (parameters()->present(name, "writeTimeScales")) {
      bool writeTimeScales = (parameters()->value(name, "writeTimeScales") != 0);
      if (writeTimeScales == getTextOutputFlag()) {
         WarnLog() << getDescription()
                   << " sets writeTimeScales, which is obsolete. Use textOutputFlag instead.\n";
      }
      else if (parameters()->present(name, "textOutputFlag")) {
         Fatal() << "writeTimeScales is obsolete as it is redundant with textOutputFlag. "
                 << getDescription() << " sets these flags to opposite values.\n";
      }
      else {
         pvAssert(writeTimeScales != getTextOutputFlag());
         Fatal() << "writeTimeScales is obsolete as it is redundant with textOutputFlag. "
                 << getDescription() << " sets writeTimeScales to "
                 << (writeTimeScales ? "true" : "false")
                 << " but the default value of textOutputFlag is "
                 << (getTextOutputFlag() ? "true" : "false") << "\n";
      }
   }
}

void AdaptiveTimeScaleProbe::ioParam_writeTimeScaleFieldnames(enum ParamsIOFlag ioFlag) {
   pvAssert(!parameters()->presentAndNotBeenRead(name, "textOutputFlag"));
   if (getTextOutputFlag()) {
      parameters()->ioParamValue(
            ioFlag,
            name,
            "writeTimeScaleFieldnames",
            &mWriteTimeScaleFieldnames,
            mWriteTimeScaleFieldnames);
   }
}

Response::Status AdaptiveTimeScaleProbe::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = ColProbe::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   mTargetProbe = message->mObjectTable->findObject<BaseProbe>(targetName);
   FatalIf(
         mTargetProbe == nullptr,
         "%s: targetName \"%s\" is not a probe in the HyPerCol.\n",
         getDescription_c(),
         targetName);
   return Response::SUCCESS;
}

Response::Status AdaptiveTimeScaleProbe::allocateDataStructures() {
   auto status = ColProbe::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   if (mTargetProbe->getNumValues() != getNumValues()) {
      if (mCommunicator->commRank() == 0) {
         Fatal() << getDescription() << ": target probe \"" << mTargetProbe->getDescription()
                 << "\" does not have the correct numValues (" << mTargetProbe->getNumValues()
                 << " instead of " << getNumValues() << ").\n";
      }
      MPI_Barrier(mCommunicator->communicator());
      exit(EXIT_FAILURE);
   }
   allocateTimeScaleController();
   return Response::SUCCESS;
}

void AdaptiveTimeScaleProbe::allocateTimeScaleController() {
   mAdaptiveTimeScaleController = new AdaptiveTimeScaleController(
         getName(),
         getNumValues(),
         mBaseMax,
         mBaseMin,
         tauFactor,
         mGrowthFactor,
         mWriteTimeScaleFieldnames,
         mCommunicator);
}

Response::Status AdaptiveTimeScaleProbe::registerData(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = ColProbe::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   mAdaptiveTimeScaleController->registerData(message);
   return Response::SUCCESS;
}

Response::Status
AdaptiveTimeScaleProbe::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   mBaseDeltaTime = message->mDeltaTime;
   return Response::SUCCESS;
}

Response::Status
AdaptiveTimeScaleProbe::respondAdaptTimestep(std::shared_ptr<AdaptTimestepMessage const> message) {
   getValues(message->mTime);
   return Response::SUCCESS;
}

// AdaptiveTimeScaleProbe::calcValues calls targetProbe->getValues() and passes the result to
// mAdaptiveTimeScaleController->calcTimesteps() to use as timeScaleTrue.
// mAdaptiveTimeScaleController->calcTimesteps() returns timeScale and copies the result into
// probeValues. AdaptiveTimeScaleProbe also processes the triggering and only reads the
// mAdaptiveTimeScaleController when triggering doesn't happen.

void AdaptiveTimeScaleProbe::calcValues(double timeValue) {
   std::vector<double> rawProbeValues;
   if (mTriggerControl != nullptr
       && mTriggerControl->needUpdate(timeValue + triggerOffset, mBaseDeltaTime)) {
      rawProbeValues.assign(getNumValues(), -1.0);
   }
   else {
      mTargetProbe->getValues(timeValue, &rawProbeValues);
   }
   pvAssert(rawProbeValues.size() == (std::size_t)getNumValues());
   // In allocateDataStructures, we checked that mTargetProbe has a compatible size.
   std::vector<double> timeSteps =
         mAdaptiveTimeScaleController->calcTimesteps(timeValue, rawProbeValues);
   memcpy(getValuesBuffer(), timeSteps.data(), sizeof(double) * getNumValues());
}

Response::Status AdaptiveTimeScaleProbe::outputState(double simTime, double deltaTime) {
   if (!mOutputStreams.empty()) {
      mAdaptiveTimeScaleController->writeTimestepInfo(simTime, mOutputStreams);
   }
   return Response::SUCCESS;
}

} /* namespace PV */
