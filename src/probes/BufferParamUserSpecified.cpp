#include "BufferParamUserSpecified.hpp"

namespace PV {

BufferParamUserSpecified::BufferParamUserSpecified(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults) {
   initialize(params, defaults);
}

void BufferParamUserSpecified::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults) {
   BufferParamInterface::initialize(params, defaults);
}

void BufferParamUserSpecified::ioParam_buffer(ParamsIOSwitch ioSwitch) {
   internal_ioParam_buffer(ioSwitch);
   if (ioSwitch == ParamsIOSwitch::Read) {
      auto bufferType = parseBufferType(getBufferString());
      setBufferType(bufferType);
   }
}

} // namespace PV
