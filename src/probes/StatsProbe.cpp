#include "StatsProbe.hpp"

#include "checkpointing/Checkpointer.hpp"
#include "columns/Communicator.hpp"
#include "columns/Messages.hpp"
#include "components/BasePublisherComponent.hpp"
#include "components/PhaseParam.hpp"
#include "include/PVLayerLoc.hpp"
#include "io/PVParams.hpp"
#include "observerpattern/BaseMessage.hpp"
#include "observerpattern/Response.hpp"
#include "probes/ProbeTriggerComponent.hpp"
#include "probes/StatsProbeTypes.hpp"
#include "utils/Timer.hpp"

#include <cMakeHeader.h>
#include <functional>

namespace PV {

StatsProbe::StatsProbe(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

StatsProbe::StatsProbe() {}

StatsProbe::~StatsProbe() {
   delete mTimerInitialization;
   delete mTimerComp;
   delete mTimerIO;
#ifdef PV_USE_MPI
   delete mTimerMPI;
#endif // PV_USE_MPI
}

void StatsProbe::assembleStatsAndOutput() {
#ifdef PV_USE_MPI
   mTimerMPI->start();
   mProbeAggregator->aggregateStoredValues(mProbeLocal->getStoredValues());
   mProbeLocal->clearStoredValues();
   mTimerMPI->stop();
#endif // PV_USE_MPI
   mTimerIO->start();
   mProbeOutputter->printGlobalStatsBuffer(mProbeAggregator->getStoredValues());
   mTimerIO->stop();

   mTimerComp->start();
   checkStats();
   mProbeAggregator->clearStoredValues();
   mTimerComp->stop();
}

// Derived classes can override checkStats() to verify that the stats satisfy desired constraints
void StatsProbe::checkStats() {}

Response::Status
StatsProbe::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = BaseObject::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   // Set target layer and trigger layer
   status = status + mProbeTargetLayer->communicateInitInfo(message);
   status = status + mProbeTrigger->communicateInitInfo(message);
   return status;
}

void StatsProbe::createComponents(char const *name, PVParams *params, Communicator const *comm) {
   // NB: the data members mName and mParams have not been set when createComponents() is called.
   createTargetLayerComponent(name, params);
   createProbeLocal(name, params);
   createProbeAggregator(name, params, comm);
   createProbeOutputter(name, params, comm);
   createProbeTrigger(name, params);
}

void StatsProbe::createProbeAggregator(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   mProbeAggregator =
         std::make_shared<StatsProbeAggregator>(name, params, comm->getLocalMPIBlock());
}

void StatsProbe::createProbeLocal(char const *name, PVParams *params) {
   mProbeLocal = std::make_shared<StatsProbeLocal>(name, params);
}

void StatsProbe::createProbeOutputter(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   mProbeOutputter = std::make_shared<StatsProbeOutputter>(name, params, comm);
}

void StatsProbe::createProbeTrigger(char const *name, PVParams *params) {
   mProbeTrigger = std::make_shared<ProbeTriggerComponent>(name, params);
}

void StatsProbe::createTargetLayerComponent(char const *name, PVParams *params) {
   mProbeTargetLayer = std::make_shared<TargetLayerComponent>(name, params);
}

void StatsProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   createComponents(name, params, comm);
   // createComponents() must be called before the base class's initialize(),
   // because BaseObject::initialize() calls the ioParamsFillGroup() method,
   // which calls each component's ioParamsFillGroup() method.
   BaseObject::initialize(name, params, comm);
}

Response::Status
StatsProbe::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   mTimerInitialization->start();
   auto status = BaseObject::initializeState(message);
   if (Response::completed(status)) {
      mProbeLocal->initializeState(getTargetLayer());
   }

   mTimerInitialization->stop();
   return Response::SUCCESS;
}

void StatsProbe::initMessageActionMap() {
   BaseObject::initMessageActionMap();
   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;

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

void StatsProbe::initProbeTimers(Checkpointer *checkpointer) {
   mTimerInitialization = new Timer(getName(), "probe", "init");
   checkpointer->registerTimer(mTimerInitialization);
   mTimerComp = new Timer(getName(), "probe", "probecomp");
   checkpointer->registerTimer(mTimerComp);
   mTimerIO = new Timer(getName(), "probe", "probeio");
#ifdef PV_USE_MPI
   mTimerMPI = new Timer(getName(), "probe", "probempi");
   checkpointer->registerTimer(mTimerMPI);
#endif // PV_USE_MPI
   checkpointer->registerTimer(mTimerIO);
}

void StatsProbe::ioParam_immediateMPIAssembly(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(
         ioFlag,
         getName(),
         "immediateMPIAssembly",
         &mImmediateMPIAssembly,
         mImmediateMPIAssembly,
         true /*warnIfAbsent*/);
}

int StatsProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseObject::ioParamsFillGroup(ioFlag);
   mProbeTargetLayer->ioParamsFillGroup(ioFlag);
   mProbeOutputter->ioParamsFillGroup(ioFlag);
   mProbeTrigger->ioParamsFillGroup(ioFlag);
   mProbeLocal->ioParamsFillGroup(ioFlag);
   mProbeAggregator->ioParamsFillGroup(ioFlag);
   ioParam_immediateMPIAssembly(ioFlag);
   return status;
}

Response::Status StatsProbe::outputState(std::shared_ptr<LayerOutputStateMessage const> message) {
   mTimerComp->start();
   if (mProbeTrigger->needUpdate(message->mTime, message->mDeltaTime)) {
      mProbeLocal->storeValues(message->mTime);
   }
   mTimerComp->stop();
   if (mImmediateMPIAssembly) {
      assembleStatsAndOutput();
   }
   return Response::SUCCESS;
}

Response::Status StatsProbe::prepareCheckpointWrite(double simTime) {
   if (!mImmediateMPIAssembly) {
      assembleStatsAndOutput();
   }
   return Response::SUCCESS;
}

Response::Status
StatsProbe::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = BaseObject::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *checkpointer = message->mDataRegistry;

   bool convertToHertz = false;
   if (mProbeLocal->getBufferType() == StatsBufferType::A) {
      convertToHertz =
            getTargetLayer()->getComponentByType<BasePublisherComponent>()->getSparseLayer();
   }
   mProbeOutputter->setConvertToHertzFlag(convertToHertz);
   mProbeOutputter->initOutputStreams(checkpointer, getTargetLayer()->getLayerLoc()->nbatch);

   initProbeTimers(checkpointer);
   return Response::SUCCESS;
}

Response::Status
StatsProbe::respondLayerOutputState(std::shared_ptr<LayerOutputStateMessage const> message) {
   auto status          = Response::SUCCESS;
   int targetLayerPhase = getTargetLayer()->getComponentByType<PhaseParam>()->getPhase();
   if (message->mPhase == targetLayerPhase) {
      status = outputState(message);
   }
   return status;
}

Response::Status
StatsProbe::respondProbeWriteParams(std::shared_ptr<ProbeWriteParamsMessage const> message) {
   writeParams();
   return Response::SUCCESS;
}

} // namespace PV
