/*
 * LayerProbe.cpp
 *
 *  Created on: Mar 7, 2009
 *      Author: rasmussn
 */

#include "LayerProbe.hpp"
#include "../layers/HyPerLayer.hpp"

namespace PV {

LayerProbe::LayerProbe() {
   initialize_base();
   // Derived classes of LayerProbe should call LayerProbe::initialize
   // themselves.
}

/**
 * @filename
 */
LayerProbe::LayerProbe(const char *name, PVParams *params, Communicator const *comm) {
   initialize_base();
   initialize(name, params, comm);
}

LayerProbe::~LayerProbe() { delete mIOTimer; }

int LayerProbe::initialize_base() {
   targetLayer = NULL;
   return PV_SUCCESS;
}

/**
 * @filename
 * @layer
 */
void LayerProbe::initialize(const char *name, PVParams *params, Communicator const *comm) {
   BaseProbe::initialize(name, params, comm);
}

void LayerProbe::initMessageActionMap() {
   BaseProbe::initMessageActionMap();
   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<LayerProbeWriteParamsMessage const>(msgptr);
      return respondLayerProbeWriteParams(castMessage);
   };
   mMessageActionMap.emplace("LayerProbeWriteParams", action);

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<LayerOutputStateMessage const>(msgptr);
      return respondLayerOutputState(castMessage);
   };
   mMessageActionMap.emplace("LayerOutputState", action);
}

void LayerProbe::ioParam_targetName(enum ParamsIOFlag ioFlag) {
   // targetLayer is a legacy parameter, so here, it's not required
   parameters()->ioParamString(
         ioFlag, name, "targetLayer", &targetName, NULL /*default*/, false /*warnIfAbsent*/);
   // But if it isn't defined, targetName must be, which BaseProbe checks for
   if (targetName == NULL) {
      BaseProbe::ioParam_targetName(ioFlag);
   }
}

Response::Status
LayerProbe::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = BaseProbe::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   // Set target layer
   targetLayer = message->mObjectTable->findObject<HyPerLayer>(targetName);
   FatalIf(
         targetLayer == nullptr,
         "%s targetLayer \"%s\" is not a layer in the column.\n",
         getDescription_c(),
         targetName);

   return Response::SUCCESS;
}

Response::Status
LayerProbe::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = BaseProbe::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }

   auto *checkpointer = message->mDataRegistry;

   mIOTimer = new Timer(getName(), "layer", "io     ");
   checkpointer->registerTimer(mIOTimer);

   return Response::SUCCESS;
}

bool LayerProbe::needRecalc(double timevalue) {
   auto *updateController = targetLayer->getComponentByType<LayerUpdateController>();
   pvAssert(updateController);
   return this->getLastUpdateTime() < updateController->getLastUpdateTime();
}

double LayerProbe::referenceUpdateTime(double simTime) const {
   auto *updateController = targetLayer->getComponentByType<LayerUpdateController>();
   pvAssert(updateController);
   return updateController->getLastUpdateTime();
}

Response::Status LayerProbe::respondLayerProbeWriteParams(
      std::shared_ptr<LayerProbeWriteParamsMessage const> message) {
   writeParams();
   return Response::SUCCESS;
}

Response::Status
LayerProbe::respondLayerOutputState(std::shared_ptr<LayerOutputStateMessage const> message) {
   Response::Status status = Response::SUCCESS;
   int targetLayerPhase    = targetLayer->getComponentByType<PhaseParam>()->getPhase();
   if (message->mPhase != targetLayerPhase) {
      return status;
   }
   status = outputStateWrapper(message->mTime, message->mDeltaTime);
   return status;
}

Response::Status LayerProbe::outputStateWrapper(double simTime, double deltaTime) {
   mIOTimer->start();
   auto status = BaseProbe::outputStateWrapper(simTime, deltaTime);
   mIOTimer->stop();
   return status;
}

} // namespace PV
