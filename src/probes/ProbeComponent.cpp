#include "ProbeComponent.hpp"

namespace PV {

ProbeComponent::ProbeComponent(std::shared_ptr<ParamsIO> paramsIO) {
   initialize(paramsIO);
}

ProbeComponent::ProbeComponent() {}

void ProbeComponent::initialize(std::shared_ptr<ParamsIO> paramsIO) {
   mParamsIO = paramsIO;
}

void ProbeComponent::setPrintParamsStream(FileStream *stream) {
   mParamsIO->setPrintParamsStream(stream);
}

void ProbeComponent::setPrintLuaStream(FileStream *stream) {
   mParamsIO->setPrintLuaStream(stream);
}

} // namespace PV
