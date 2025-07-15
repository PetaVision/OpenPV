#include "BufferParamVMembraneSpecified.hpp"
#include "probes/BufferParamInterface.hpp"
#include "probes/StatsProbeTypes.hpp"
#include "utils/PVLog.hpp"

namespace PV {

BufferParamVMembraneSpecified::BufferParamVMembraneSpecified(std::shared_ptr<ParamsIO> paramsIO) {
   initialize(paramsIO);
}

BufferParamVMembraneSpecified::~BufferParamVMembraneSpecified() {}

void BufferParamVMembraneSpecified::initialize(std::shared_ptr<ParamsIO> paramsIO) {
   BufferParamInterface::initialize(paramsIO);
}

void BufferParamVMembraneSpecified::ioParam_buffer(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read) {
      if (mParamsIO->isPresent("buffer")) {
         mParamsIO->handleUnnecessaryParameter("buffer");
         std::string bufferString;
         mParamsIO->ioParam(ioSwitch, "buffer", &bufferString);
         auto bufferType = parseBufferType(bufferString);
         FatalIf(
               bufferType != StatsBufferType::V,
               "Probe %s buffer parameter \"%s\" is inconsistent with allowed values "
               "\"MembranePotential\" or \"V\"\n",
               getName_c(),
               bufferString.c_str());
      }
      setBufferType(StatsBufferType::V);
   }
}

} // namespace PV
