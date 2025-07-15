#include "ResetStateOnTriggerTestProbe.hpp"
#include <arch/mpi/mpi.h>
#include <columns/BaseObject.hpp>
#include <columns/Communicator.hpp>
#include <columns/Messages.hpp>
#include <components/BasePublisherComponent.hpp>
#include <components/PhaseParam.hpp>
#include <params/PVParams.hpp>
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

ResetStateOnTriggerTestProbe::ResetStateOnTriggerTestProbe(std::shared_ptr<PV::ParamsIO> paramsIO, PV::Communicator const *comm) {
   initialize(paramsIO, comm);
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

void ResetStateOnTriggerTestProbe::initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   mProbeLocal         = std::make_shared<ResetStateOnTriggerTestProbeLocal>(paramsIO);
   mTargetLayerLocator = std::make_shared<TargetLayerComponent>(paramsIO);
   mProbeOutputter =
         std::make_shared<ResetStateOnTriggerTestProbeOutputter>(paramsIO, comm);
   BaseObject::initialize(paramsIO, comm);
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

int ResetStateOnTriggerTestProbe::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = BaseObject::ioParamsFillGroup(ioSwitch);
   mTargetLayerLocator->ioParamsFillGroup(ioSwitch);
   mProbeOutputter->ioParamsFillGroup(ioSwitch);
   mProbeLocal->ioParamsFillGroup(ioSwitch);
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

PV::Response::Status ResetStateOnTriggerTestProbe::registerData(
      std::shared_ptr<PV::RegisterDataMessage<PV::Checkpointer> const> message) {
   auto status = PV::BaseObject::registerData(message);
   if (!PV::Response::completed(status)) {
      return status;
   }
   auto *targetLayer = mTargetLayerLocator->getTargetLayer();
   mProbeLocal->initializeState(targetLayer);

   mTargetLayerLoc = targetLayer->getLayerLoc();

   auto *targetPublisher = targetLayer->getComponentByType<PV::BasePublisherComponent>();
   mTargetLayerData      = targetPublisher->getLayerData();

   auto *checkpointer = message->mDataRegistry;
   mProbeOutputter->initOutputStreams(checkpointer, mTargetLayerLoc->nbatch);
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
   mParamsIO->setPrintParamsStream(message->mPrintParamsStream);
   mParamsIO->setPrintLuaStream(message->mPrintLuaStream);
   return PV::Response::SUCCESS;
}

BaseObject *
createResetStateOnTriggerTestProbe(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) {
   return new ResetStateOnTriggerTestProbe(paramsIO, comm);
}
