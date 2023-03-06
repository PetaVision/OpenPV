#include "AbstractNormProbe.hpp"
#include "components/PhaseParam.hpp"
#include "observerpattern/BaseMessage.hpp"
#include "utils/PVAssert.hpp"
#include <functional>

namespace PV {

AbstractNormProbe::AbstractNormProbe(char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

Response::Status AbstractNormProbe::allocateDataStructures() {
   mLocalNBatch = getTargetLayer()->getLayerLoc()->nbatch;
   setNumValues(mLocalNBatch);
   if (mEnergyProbeComponent) {
      setCoefficient(mEnergyProbeComponent->getCoefficient());
   }
   return Response::SUCCESS;
}

void AbstractNormProbe::calcValues(double timestamp) {
   mProbeLocal->storeValues(timestamp);

   mProbeAggregator->aggregateStoredValues(mProbeLocal->getStoredValues());
   mProbeLocal->clearStoredValues();

   auto const &storedValues = mProbeAggregator->getStoredValues();
   auto bufferSize          = storedValues.size();
   pvAssert(bufferSize > static_cast<batchwidth_type>(0));
   auto lastDataIndex             = bufferSize - static_cast<batchwidth_type>(1);
   LayerProbeData const &lastData = storedValues.getData(lastDataIndex);
   setValues(lastData);
}

Response::Status
AbstractNormProbe::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = ProbeInterface::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   // Set target layer
   status = status + mProbeTargetLayer->communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   // Set up triggering
   status = status + mProbeTrigger->communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   // Add probe to energy probe if there is one
   status = status + mEnergyProbeComponent->communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *energyProbe = mEnergyProbeComponent->getEnergyProbe();
   if (energyProbe and !mAddedToEnergyProbe) {
      energyProbe->addTerm(this);
      status              = Response::SUCCESS;
      mAddedToEnergyProbe = true;
   }
   return status;
}

void AbstractNormProbe::createComponents(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   // NB: the data members mName and mParams have not been set when createComponents() is called.
   createTargetLayerComponent(name, params);
   createProbeLocal(name, params);
   createProbeAggregator(name, params, comm);
   createProbeOutputter(name, params, comm);
   createProbeTrigger(name, params);
   createEnergyProbeComponent(name, params);
}

void AbstractNormProbe::createEnergyProbeComponent(char const *name, PVParams *params) {
   mEnergyProbeComponent = std::make_shared<EnergyProbeComponent>(name, params);
}

void AbstractNormProbe::createProbeAggregator(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   mProbeAggregator = std::make_shared<NormProbeAggregator>(name, params, comm->getLocalMPIBlock());
}

void AbstractNormProbe::createProbeOutputter(
      char const *name,
      PVParams *params,
      Communicator const *comm) {
   mProbeOutputter = std::make_shared<NormProbeOutputter>(name, params, comm);
}

void AbstractNormProbe::createProbeTrigger(char const *name, PVParams *params) {
   mProbeTrigger = std::make_shared<ProbeTriggerComponent>(name, params);
}

void AbstractNormProbe::createTargetLayerComponent(char const *name, PVParams *params) {
   mProbeTargetLayer = std::make_shared<TargetLayerComponent>(name, params);
}

void AbstractNormProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   createComponents(name, params, comm);
   // createComponents() must be called before the base class's initialize(),
   // because BaseObject::initialize() calls the ioParamsFillGroup() method,
   // which calls each component's ioParamsFillGroup() method.
   ProbeInterface::initialize(name, params, comm);
}

Response::Status
AbstractNormProbe::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   auto status = ProbeInterface::initializeState(message);
   if (Response::completed(status)) {
      mProbeLocal->initializeState(getTargetLayer());
      mEnergyProbeComponent->initializeState(getTargetLayer());
   }

   return Response::SUCCESS;
}

void AbstractNormProbe::initMessageActionMap() {
   ProbeInterface::initMessageActionMap();
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

int AbstractNormProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ProbeInterface::ioParamsFillGroup(ioFlag);
   mProbeTargetLayer->ioParamsFillGroup(ioFlag);
   mProbeOutputter->ioParamsFillGroup(ioFlag);
   mProbeTrigger->ioParamsFillGroup(ioFlag);
   mProbeLocal->ioParamsFillGroup(ioFlag);
   mProbeAggregator->ioParamsFillGroup(ioFlag);
   mEnergyProbeComponent->ioParamsFillGroup(ioFlag);
   return status;
}

Response::Status
AbstractNormProbe::outputState(std::shared_ptr<LayerOutputStateMessage const> message) {
   if (mProbeTrigger->needUpdate(message->mTime, message->mDeltaTime)) {
      getValues(message->mTime);
   }

   return Response::SUCCESS;
}

Response::Status AbstractNormProbe::prepareCheckpointWrite(double simTime) {
   mProbeOutputter->printGlobalNormsBuffer(
         mProbeAggregator->getStoredValues(), getTargetLayer()->getNumGlobalNeurons());
   mProbeAggregator->clearStoredValues();
   return Response::SUCCESS;
}

Response::Status AbstractNormProbe::processCheckpointRead(double simTime) {
   // This assumes that if there is a trigger layer, the layer triggered
   // at the time of the checkpoint being read from; or at least that
   // the target layer has not updated since then.
   getValues(simTime);

   // calcValues() uses ProbeAggregator to compute the values in ProbeInterface::mValues,
   // but writing the checkpoint clears the ProbeAggregator object, so
   // on exit from reading a checkpoint, the ProbeAggreagotr should be cleared.
   mProbeAggregator->clearStoredValues();

   return Response::SUCCESS;
}

Response::Status
AbstractNormProbe::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = ProbeInterface::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *checkpointer = message->mDataRegistry;
   mProbeOutputter->initOutputStreams(checkpointer, mLocalNBatch);
   return Response::SUCCESS;
}

Response::Status
AbstractNormProbe::respondLayerOutputState(std::shared_ptr<LayerOutputStateMessage const> message) {
   auto status          = Response::SUCCESS;
   int targetLayerPhase = getTargetLayer()->getComponentByType<PhaseParam>()->getPhase();
   if (message->mPhase == targetLayerPhase) {
      status = outputState(message);
   }
   return status;
}

Response::Status
AbstractNormProbe::respondProbeWriteParams(std::shared_ptr<ProbeWriteParamsMessage const> message) {
   writeParams();
   return Response::SUCCESS;
}

} // namespace PV
