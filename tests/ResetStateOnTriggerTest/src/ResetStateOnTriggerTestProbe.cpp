#include "ResetStateOnTriggerTestProbe.hpp"
#include <arch/mpi/mpi.h>
#include <columns/BaseObject.hpp>
#include <columns/Communicator.hpp>
#include <columns/Messages.hpp>
#include <components/BasePublisherComponent.hpp>
#include <components/PhaseParam.hpp>
#include <io/PVParams.hpp>
#include <layers/HyPerLayer.hpp>
#include <observerpattern/BaseMessage.hpp>
#include <observerpattern/Response.hpp>
#include <probes/ProbeData.hpp>
#include <probes/TargetLayerComponent.hpp>
#include <utils/PVAssert.hpp>

#include <cmath>
#include <functional>
#include <vector>

using PV::BaseMessage;

ResetStateOnTriggerTestProbe::ResetStateOnTriggerTestProbe(
      char const *name,
      PV::PVParams *params,
      PV::Communicator const *comm) {
   initialize(name, params, comm);
}

ResetStateOnTriggerTestProbe::~ResetStateOnTriggerTestProbe() {}

PV::Response::Status ResetStateOnTriggerTestProbe::communicateInitInfo(
      std::shared_ptr<PV::CommunicateInitInfoMessage const> message) {
   auto status = PV::BaseObject::communicateInitInfo(message);
   if (!PV::Response::completed(status)) {
      return status;
   }
   status = status + mTargetLayerLocator->communicateInitInfo(message);
   return PV::Response::SUCCESS;
}

void ResetStateOnTriggerTestProbe::initialize(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   mProbeLocal         = std::make_shared<ResetStateOnTriggerTestProbeLocal>(name, params);
   mTargetLayerLocator = std::make_shared<TargetLayerComponent>(name, params);
   mProbeOutputter = std::make_shared<ResetStateOnTriggerTestProbeOutputter>(name, params, comm);
   BaseObject::initialize(name, params, comm);
}

PV::Response::Status ResetStateOnTriggerTestProbe::initializeState(
      std::shared_ptr<InitializeStateMessage const> message) {
   auto status = BaseObject::initializeState(message);
   if (!PV::Response::completed(status)) {
      return status;
   }
   auto *targetLayer = mTargetLayerLocator->getTargetLayer();
   pvAssert(targetLayer);
   if (PV::Response::completed(status)) {
      mProbeLocal->initializeState(targetLayer);
      auto *targetLayer     = mTargetLayerLocator->getTargetLayer();
      auto *targetPublisher = targetLayer->getComponentByType<PV::BasePublisherComponent>();
      mTargetLayerLoc       = targetLayer->getLayerLoc();
      mTargetLayerData      = targetPublisher->getLayerData();
   }
   return PV::Response::Status::SUCCESS;
}

void ResetStateOnTriggerTestProbe::initMessageActionMap() {
   BaseObject::initMessageActionMap();
   std::function<PV::Response::Status(std::shared_ptr<BaseMessage const>)> action;

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<LayerOutputStateMessage const>(msgptr);
      return respondLayerOutputState(castMessage);
   };
   mMessageActionMap.emplace("LayerOutputState", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<ProbeWriteParamsMessage const>(msgptr);
      return respondProbeWriteParams(castMessage);
   };
   mMessageActionMap.emplace("ProbeWriteParams", action);
}

int ResetStateOnTriggerTestProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseObject::ioParamsFillGroup(ioFlag);
   mTargetLayerLocator->ioParamsFillGroup(ioFlag);
   mProbeOutputter->ioParamsFillGroup(ioFlag);
   mProbeLocal->ioParamsFillGroup(ioFlag);
   return status;
}

PV::Response::Status
ResetStateOnTriggerTestProbe::outputState(std::shared_ptr<LayerOutputStateMessage const> message) {
   double timestepNum = std::nearbyint(message->mTime / message->mDeltaTime);

   if (timestepNum > 0.0) {
      mProbeLocal->clearStoredValues();
      mProbeLocal->storeValues(timestepNum);
      pvAssert(static_cast<int>(mProbeLocal->getStoredValues().size()) == 1);
      ProbeData<int> const &localDiscrepsData     = mProbeLocal->getStoredValues().getData(0);
      std::vector<int> const &localDiscrepsVector = localDiscrepsData.getValues();
      int nBatch                                  = static_cast<int>(localDiscrepsVector.size());
      ProbeData<int> globalDiscreps(message->mTime, nBatch);
      MPI_Allreduce(
            localDiscrepsVector.data(),
            &globalDiscreps.getValue(0),
            nBatch,
            MPI_INT,
            MPI_SUM,
            mCommunicator->communicator());
      mProbeOutputter->printGlobalStatsBuffer(globalDiscreps);
   }
   return PV::Response::SUCCESS;
}

PV::Response::Status ResetStateOnTriggerTestProbe::respondLayerOutputState(
      std::shared_ptr<LayerOutputStateMessage const> message) {
   auto status          = PV::Response::SUCCESS;
   auto *targetLayer    = mTargetLayerLocator->getTargetLayer();
   int targetLayerPhase = targetLayer->getComponentByType<PV::PhaseParam>()->getPhase();
   if (message->mPhase == targetLayerPhase) {
      status = outputState(message);
   }
   return status;
}

PV::Response::Status ResetStateOnTriggerTestProbe::respondProbeWriteParams(
      std::shared_ptr<ProbeWriteParamsMessage const> message) {
   writeParams();
   return PV::Response::SUCCESS;
}

BaseObject *
createResetStateOnTriggerTestProbe(char const *name, PVParams *params, Communicator const *comm) {
   return new ResetStateOnTriggerTestProbe(name, params, comm);
}
