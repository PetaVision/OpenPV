#include "AbstractNormProbe.hpp"
#include "components/PhaseParam.hpp"
#include "observerpattern/BaseMessage.hpp"
#include "utils/PVAssert.hpp"
#include <functional>

namespace PV {

AbstractNormProbe::AbstractNormProbe(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
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
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   // NB: the data members mName and mParams have not been set when createComponents() is called.
   createTargetLayerComponent(params, defaults);
   createProbeLocal(params, defaults);
   createProbeAggregator(params, defaults, comm);
   createProbeOutputter(params, defaults, comm);
   createProbeTrigger(params, defaults);
   createEnergyProbeComponent(params, defaults);
}

void AbstractNormProbe::createEnergyProbeComponent(
      std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   mEnergyProbeComponent = std::make_shared<EnergyProbeComponent>(params, defaults);
}

void AbstractNormProbe::createProbeAggregator(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   mProbeAggregator = std::make_shared<NormProbeAggregator>(params, defaults, comm->getLocalMPIBlock());
}

void AbstractNormProbe::createProbeOutputter(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   mProbeOutputter = std::make_shared<NormProbeOutputter>(params, defaults, comm);
}

void AbstractNormProbe::createProbeTrigger(
      std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   mProbeTrigger = std::make_shared<ProbeTriggerComponent>(params, defaults);
}

void AbstractNormProbe::createTargetLayerComponent(
      std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   mProbeTargetLayer = std::make_shared<TargetLayerComponent>(params, defaults);
}

void AbstractNormProbe::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   createComponents(params, defaults, comm);
   // createComponents() must be called before the base class's initialize(),
   // because BaseObject::initialize() calls the ioParamsFillGroup() method,
   // which calls each component's ioParamsFillGroup() method.
   ProbeInterface::initialize(params, defaults, comm);
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
}

int AbstractNormProbe::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = ProbeInterface::ioParamsFillGroup(ioSwitch);
   mProbeTargetLayer->ioParamsFillGroup(ioSwitch);
   mProbeOutputter->ioParamsFillGroup(ioSwitch);
   mProbeTrigger->ioParamsFillGroup(ioSwitch);
   mProbeLocal->ioParamsFillGroup(ioSwitch);
   mProbeAggregator->ioParamsFillGroup(ioSwitch);
   mEnergyProbeComponent->ioParamsFillGroup(ioSwitch);
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

void AbstractNormProbe::setPrintStreams(FileStream *printParamsStream, FileStream *printLuaStream) {
   ProbeInterface::setPrintStreams(printParamsStream, printLuaStream);

   setComponentPrintStreams(*mProbeTargetLayer, printParamsStream, printLuaStream);
   setComponentPrintStreams(*mProbeLocal, printParamsStream, printLuaStream);
   setComponentPrintStreams(*mProbeAggregator, printParamsStream, printLuaStream);
   setComponentPrintStreams(*mProbeOutputter, printParamsStream, printLuaStream);
   setComponentPrintStreams(*mProbeTrigger, printParamsStream, printLuaStream);
   setComponentPrintStreams(*mEnergyProbeComponent, printParamsStream, printLuaStream);
}

} // namespace PV
