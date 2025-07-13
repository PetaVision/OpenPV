#include "ProbeComponent.hpp"

namespace PV {

ProbeComponent::ProbeComponent(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   initialize(params, defaults);
}

ProbeComponent::ProbeComponent() {}

void ProbeComponent::initialize(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   mParamsIO = std::make_shared<ParamsIO>(params, defaults);
}

void ProbeComponent::setPrintParamsStream(FileStream *stream) {
   mParamsIO->setPrintParamsStream(stream);
}

void ProbeComponent::setPrintLuaStream(FileStream *stream) {
   mParamsIO->setPrintLuaStream(stream);
}

} // namespace PV
