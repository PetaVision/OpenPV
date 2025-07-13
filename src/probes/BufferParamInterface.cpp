#include "BufferParamInterface.hpp"
#include "utils/PVLog.hpp"
#include <cctype>
#include <cstdlib>
#include <cstring>

namespace PV {

BufferParamInterface::~BufferParamInterface() {}

void BufferParamInterface::initialize(
      std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) {
   ProbeComponent::initialize(params, defaults);
}

void BufferParamInterface::internal_ioParam_buffer(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "buffer", &mBufferString);
}

StatsBufferType BufferParamInterface::parseBufferType(std::string const &bufferString) {
   std::string buffer(bufferString);
   StatsBufferType bufferType;
   for (size_t c = 0; c < buffer.size(); c++) {
      buffer[c] = (char)tolower((int)buffer[c]);
   }
   if (buffer == "v" or buffer == "membranepotential") {
      bufferType = StatsBufferType::V;
   }
   else if (buffer == "a" or buffer == "activity") {
      bufferType = StatsBufferType::A;
   }
   else {
      Fatal().printf(
            "Probe %s buffer type \"%s\" is not recognized.\n", getName_c(), bufferString.c_str());
   }
   return bufferType;
}

void BufferParamInterface::setBufferType(StatsBufferType bufferType) {
   mBufferType = bufferType;
   switch (bufferType) {
      case StatsBufferType::A: mBufferString = "Activity"; break;
      case StatsBufferType::V: mBufferString = "MembranePotential"; break;
      default: Fatal().printf("Unrecognized StatsBufferType in probe %s\n", getName_c());
   }
}

} // namespace PV
