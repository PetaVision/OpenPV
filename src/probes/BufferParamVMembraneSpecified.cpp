#include "BufferParamVMembraneSpecified.hpp"
#include "probes/BufferParamInterface.hpp"
#include "probes/StatsProbeTypes.hpp"
#include "utils/PVLog.hpp"

namespace PV {

BufferParamVMembraneSpecified::BufferParamVMembraneSpecified(char const *name, PVParams *params) {
   initialize(name, params);
}

BufferParamVMembraneSpecified::~BufferParamVMembraneSpecified() {}

void BufferParamVMembraneSpecified::initialize(char const *name, PVParams *params) {
   BufferParamInterface::initialize(name, params);
}

void BufferParamVMembraneSpecified::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      if (getParams()->stringPresent(getName_c(), "buffer")) {
         getParams()->handleUnnecessaryStringParameter(getName_c(), "buffer");
         char const *bufferString = getParams()->stringValue(getName_c(), "buffer");
         auto bufferType          = parseBufferType(bufferString);
         FatalIf(
               bufferType != StatsBufferType::V,
               "Probe %s buffer parameter \"%s\" is inconsistent with allowed values "
               "\"MembranePotential\" or \"V\"\n",
               getName_c(),
               bufferString);
      }
      setBufferType(StatsBufferType::V);
   }
}

} // namespace PV
