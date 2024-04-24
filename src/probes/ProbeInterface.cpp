#include "ProbeInterface.hpp"
#include "utils/PVAssert.hpp"
#include <memory>

namespace PV {

ProbeInterface::ProbeInterface(const char *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

std::vector<double> const &ProbeInterface::getValues() const { return mValues->getValues(); }

std::vector<double> const &ProbeInterface::getValues(double timestamp) {
   if (timestamp > mValues->getTimestamp()) {
      calcValues(timestamp);
   }
   return getValues();
}

void ProbeInterface::initialize(const char *name, PVParams *params, Communicator const *comm) {
   BaseObject::initialize(name, params, comm);
}

void ProbeInterface::initMessageActionMap() {
   BaseObject::initMessageActionMap();
   std::function<Response::Status(std::shared_ptr<BaseMessage const>)> action;

   action = [this](std::shared_ptr<BaseMessage const> msgptr) {
      auto castMessage = std::dynamic_pointer_cast<ProbeWriteParamsMessage const>(msgptr);
      return respondProbeWriteParams(castMessage);
   };
   mMessageActionMap.emplace("ProbeWriteParams", action);
}

Response::Status
ProbeInterface::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = BaseObject::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   return Response::SUCCESS;
}

Response::Status
ProbeInterface::respondProbeWriteParams(std::shared_ptr<ProbeWriteParamsMessage const> message) {
   writeParams();
   return Response::SUCCESS;
}

void ProbeInterface::setNumValues(int numValues) {
   mNumValues = numValues;
   mValues    = std::make_shared<ProbeData<double>>(-1.0, numValues);
   // Use a negative timestamp so that the first time getValues(double timestamp) is called,
   // even if timestamp is 0.0, the value of the probe gets calculated.
}

void ProbeInterface::setValues(ProbeData<double> const &newValues) {
   setValues(newValues.getTimestamp(), newValues.getValues());
}

void ProbeInterface::setValues(double timestamp, std::vector<double> const &newValues) {
   pvAssert(newValues.size() == mValues->size());
   mValues->reset(timestamp, newValues);
}

} // namespace PV
