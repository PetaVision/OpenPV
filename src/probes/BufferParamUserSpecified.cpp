#include "BufferParamUserSpecified.hpp"

namespace PV {

BufferParamUserSpecified::BufferParamUserSpecified(std::shared_ptr<ParamsIO> paramsIO) {
   initialize(paramsIO);
}

void BufferParamUserSpecified::initialize(std::shared_ptr<ParamsIO> paramsIO) {
   BufferParamInterface::initialize(paramsIO);
}

void BufferParamUserSpecified::ioParam_buffer(ParamsIOSwitch ioSwitch) {
   internal_ioParam_buffer(ioSwitch);
   if (ioSwitch == ParamsIOSwitch::Read) {
      auto bufferType = parseBufferType(getBufferString());
      setBufferType(bufferType);
   }
}

} // namespace PV
