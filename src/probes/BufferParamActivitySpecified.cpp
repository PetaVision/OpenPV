#include "BufferParamActivitySpecified.hpp"
#include "probes/BufferParamInterface.hpp"
#include "probes/StatsProbeTypes.hpp"
#include "utils/PVLog.hpp"

namespace PV {

BufferParamActivitySpecified::BufferParamActivitySpecified(char const *name, PVParams *params) {
   initialize(name, params);
}

BufferParamActivitySpecified::~BufferParamActivitySpecified() {}

void BufferParamActivitySpecified::initialize(char const *name, PVParams *params) {
   BufferParamInterface::initialize(name, params);
}

void BufferParamActivitySpecified::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      if (getParams()->stringPresent(getName_c(), "buffer")) {
         getParams()->handleUnnecessaryStringParameter(getName_c(), "buffer");
         char const *bufferString = getParams()->stringValue(getName_c(), "buffer");
         auto bufferType          = parseBufferType(bufferString);
         FatalIf(
               bufferType != StatsBufferType::A,
               "Probe %s buffer parameter \"%s\" is inconsistent with allowed values "
               "\"Activity\" or \"A\"\n",
               getName_c(),
               bufferString);
      }
      setBufferType(StatsBufferType::A);
   }
}

} // namespace PV
