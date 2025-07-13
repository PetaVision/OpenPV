#include "BufferParamActivitySpecified.hpp"
#include "probes/BufferParamInterface.hpp"
#include "probes/StatsProbeTypes.hpp"
#include "utils/PVLog.hpp"

namespace PV {

BufferParamActivitySpecified::BufferParamActivitySpecified(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults) {
   initialize(params, defaults);
}

BufferParamActivitySpecified::~BufferParamActivitySpecified() {}

void BufferParamActivitySpecified::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults) {
   BufferParamInterface::initialize(params, defaults);
}

void BufferParamActivitySpecified::ioParam_buffer(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read) {
      if (mParamsIO->isPresent("buffer")) {
         mParamsIO->handleUnnecessaryParameter("buffer");
         std::string bufferString;
         mParamsIO->ioParam(ioSwitch, "buffer", &bufferString);
         auto bufferType = parseBufferType(bufferString);
         FatalIf(
               bufferType != StatsBufferType::A,
               "Probe %s buffer parameter \"%s\" is inconsistent with allowed values "
               "\"Activity\" or \"A\"\n",
               getName_c(),
               bufferString.c_str());
      }
      setBufferType(StatsBufferType::A);
   }
}

} // namespace PV
