#include "BufferParamInterface.hpp"
#include "utils/PVLog.hpp"
#include <cctype>
#include <cstdlib>
#include <cstring>

namespace PV {

BufferParamInterface::~BufferParamInterface() { free(mBufferString); }

void BufferParamInterface::initialize(char const *name, PVParams *params) {
   ProbeComponent::initialize(name, params);
}

void BufferParamInterface::internal_ioParam_buffer(enum ParamsIOFlag ioFlag) {
   getParams()->ioParamString(
         ioFlag, getName_c(), "buffer", &mBufferString, "Activity", true /*warnIfAbsent*/);
}

StatsBufferType BufferParamInterface::parseBufferType(char const *bufferString) {
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
            "Probe %s buffer type \"%s\" is not recognized.\n", getName_c(), buffer.c_str());
   }
   return bufferType;
}

void BufferParamInterface::setBufferType(StatsBufferType bufferType) {
   mBufferType = bufferType;
   free(mBufferString);
   switch (bufferType) {
      case StatsBufferType::A: mBufferString = strdup("Activity"); break;
      case StatsBufferType::V: mBufferString = strdup("MembranePotential"); break;
      default: Fatal().printf("Unrecognized StatsBufferType in probe %s\n", getName_c());
   }
}

} // namespace PV
