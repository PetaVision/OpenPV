#include "BufferParamUserSpecified.hpp"

namespace PV {

BufferParamUserSpecified::BufferParamUserSpecified(char const *name, PVParams *params) {
   initialize(name, params);
}

void BufferParamUserSpecified::initialize(char const *name, PVParams *params) {
   BufferParamInterface::initialize(name, params);
}

void BufferParamUserSpecified::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   internal_ioParam_buffer(ioFlag);
   if (ioFlag == PARAMS_IO_READ) {
      auto bufferType = parseBufferType(getBufferString());
      setBufferType(bufferType);
   }
}

} // namespace PV
