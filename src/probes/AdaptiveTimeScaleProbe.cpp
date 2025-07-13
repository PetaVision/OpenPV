/*
 * AdaptiveTimeScaleProbe.cpp
 *
 *  Created on: Aug 18, 2016
 *      Author: pschultz
 */

#include "AdaptiveTimeScaleProbe.hpp"
#include "columns/Messages.hpp"
#include "observerpattern/BaseMessage.hpp"
#include "probes/AdaptiveTimeScaleProbeOutputter.hpp"
#include "probes/ProbeTriggerComponent.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include <cstdlib>
#include <functional>
#include <vector>

namespace PV {

AdaptiveTimeScaleProbe::AdaptiveTimeScaleProbe(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

AdaptiveTimeScaleProbe::AdaptiveTimeScaleProbe() {}

AdaptiveTimeScaleProbe::~AdaptiveTimeScaleProbe() {
    delete mAdaptiveTimeScaleController;
}

void AdaptiveTimeScaleProbe::createComponents(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   // NB: the data members mName and mParams have not been set when createComponents() is called.
   createProbeOutputter(params, defaults, comm);
   createProbeTrigger(params, defaults);
}

void AdaptiveTimeScaleProbe::createProbeOutputter(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   mProbeOutputter = std::make_shared<AdaptiveTimeScaleProbeOutputter>(params, defaults, comm);
}

void AdaptiveTimeScaleProbe::createProbeTrigger(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults) {
   mProbeTrigger = std::make_shared<ProbeTriggerComponent>(params, defaults);
}

void AdaptiveTimeScaleProbe::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   createComponents(params, defaults, comm);
   // createComponents() must be called before the base class's initialize(),
   // because BaseObject::initialize() calls the ioParamsFillGroup() method,
   // which calls each component's ioParamsFillGroup() method.
   ProbeInterface::initialize(params, defaults, comm);
}

void AdaptiveTimeScaleProbe::initMessageActionMap() {
   ProbeInterface::initMessageActionMap();
   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<AdaptTimestepMessage const>(msgptr);
      return respondAdaptTimestep(castMessage);
   };
   mMessageActionMap.emplace("AdaptTimestep", action);
}

int AdaptiveTimeScaleProbe::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = ProbeInterface::ioParamsFillGroup(ioSwitch);
   ioParam_targetName(ioSwitch);
   mProbeOutputter->ioParamsFillGroup(ioSwitch);
   mProbeTrigger->ioParamsFillGroup(ioSwitch);
   ioParam_baseMax(ioSwitch);
   ioParam_baseMin(ioSwitch);
   ioParam_tauFactor(ioSwitch);
   ioParam_growthFactor(ioSwitch);
   return status;
}

void AdaptiveTimeScaleProbe::ioParam_targetName(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "targetName", &mTargetName);
}

void AdaptiveTimeScaleProbe::ioParam_baseMax(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "baseMax", &mBaseMax);
}

void AdaptiveTimeScaleProbe::ioParam_baseMin(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "baseMin", &mBaseMin);
}

void AdaptiveTimeScaleProbe::ioParam_tauFactor(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "tauFactor", &tauFactor);
}

void AdaptiveTimeScaleProbe::ioParam_growthFactor(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "growthFactor", &mGrowthFactor);
}

Response::Status AdaptiveTimeScaleProbe::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = ProbeInterface::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   mTargetProbe = message->mObjectTable->findObject<ProbeInterface>(mTargetName);
   FatalIf(
         mTargetProbe == nullptr,
         "%s: targetName \"%s\" is not a suitable probe type.\n",
         getDescription_c(),
         mTargetName.c_str());

   // Set up triggering
   status = status + mProbeTrigger->communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   return Response::SUCCESS;
}

Response::Status AdaptiveTimeScaleProbe::allocateDataStructures() {
   auto status = ProbeInterface::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   if (!mTargetProbe->getDataStructuresAllocatedFlag()) {
      InfoLog().printf(
            "%s must postpone until %s allocates.\n",
            getDescription_c(),
            mTargetProbe->getDescription_c());
      return Response::POSTPONE;
   }
   int batchWidth = mTargetProbe->getNumValues();
   setNumValues(batchWidth);
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
         mCommunicator);
}

Response::Status AdaptiveTimeScaleProbe::registerData(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = ProbeInterface::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   mAdaptiveTimeScaleController->registerData(message);
   auto *checkpointer = message->mDataRegistry;
   mProbeOutputter->initOutputStreams(checkpointer, getNumValues());
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

void AdaptiveTimeScaleProbe::calcValues(double timestamp) {
   std::vector<double> probeValues;
   if (mProbeTrigger and mProbeTrigger->needUpdate(timestamp, mBaseDeltaTime)) {
      probeValues.assign(getNumValues(), -1.0);
   }
   else {
      // Since AdaptTimestep is called at the beginning of the timestep, we don't want to cause
      // the downstream probes to recalculate here, since the layers haven't updated for the
      // current timestep yet. Hence, we call TargetProbe->getValues() with no argument.
      probeValues = mTargetProbe->getValues();

      // In allocateDataStructures, we set NumValues to the target probe's NumValues.
      pvAssert(getNumValues() == static_cast<int>(probeValues.size()));
   }
   std::vector<TimeScaleData> const &newTimeScales =
         mAdaptiveTimeScaleController->calcTimesteps(probeValues);
   ProbeData<TimeScaleData> newTimeScaleData(timestamp, getNumValues());
   for (int b = 0; b < getNumValues(); ++b) {
       newTimeScaleData.getValue(b) = newTimeScales[b];
       probeValues[b] = newTimeScales[b].mTimeScale;
   }
   mStoredValues.store(newTimeScaleData);
   setValues(timestamp, probeValues);
}

Response::Status AdaptiveTimeScaleProbe::prepareCheckpointWrite(double simTime) {
   mProbeOutputter->printTimeScaleBuffer(mStoredValues);
   mStoredValues.clear();
   return Response::SUCCESS;
}

void AdaptiveTimeScaleProbe::setPrintStreams(
      FileStream *printParamsStream, FileStream *printLuaStream) {
   ProbeInterface::setPrintStreams(printParamsStream, printLuaStream);
   setComponentPrintStreams(*mProbeOutputter, printParamsStream, printLuaStream);
   setComponentPrintStreams(*mProbeTrigger, printParamsStream, printLuaStream);
}

} // namespace PV
