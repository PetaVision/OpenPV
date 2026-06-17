#include <cassert>
#include "TimerLayer.hpp"
#include "components/TimerLayerUpdateController.hpp"

namespace PV {

TimerLayer::TimerLayer(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

TimerLayer::~TimerLayer() {}

void TimerLayer::initialize(const char *name, PVParams *params, Communicator const *comm) {
   HyPerLayer::initialize(name, params, comm);
}

void TimerLayer::fillComponentTable() {
   // Deliberately do not call HyPerLayer::fillComponentTable() since many components are absent
   mLayerUpdateController = createLayerUpdateController();
   if (mLayerUpdateController) {
      addUniqueComponent(mLayerUpdateController);
   }
}

void TimerLayer::initMessageActionMap() {
   // Deliberately bypass HyPerLayer::initMessageActionMap() since many messages are inapplicable
   ComponentBasedObject::initMessageActionMap();
   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;


   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<LayerUpdateStateMessage const>(msgptr);
      return respondLayerUpdateState(castMessage);
   };
   mMessageActionMap.emplace("LayerUpdateState", action);
}

LayerUpdateController *TimerLayer::createLayerUpdateController() {
   return new TimerLayerUpdateController(getName(), parameters(), mCommunicator);
}

Response::Status
TimerLayer::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   assert(mLayerUpdateController);
   mLayerUpdateController->respond(message);
   return Response::SUCCESS;
}

} // namespace PV
