#include "L1NormLCAProbeLocal.hpp"

namespace PV {

L1NormLCAProbeLocal::L1NormLCAProbeLocal(
      char const *objName,
      PVParams *params) {
   initialize(objName, params);
}

void L1NormLCAProbeLocal::initialize(char const *objName, PVParams *params) {
   L1NormProbeLocal::initialize(objName, params);
}

void L1NormLCAProbeLocal::warnUnnecessaryParameter(char const *paramName) {
   if (getParams()->present(getName_c(), paramName)) {
      char const *className = getParams()->groupKeywordFromName(getName_c());
      WarnLog().printf(
            "Parameter %s is present in the params file for %s \"%s\", but %s does not use it. "
            "Instead, %s is taken from the target layer.\n",
            paramName,
            className,
            getName_c(),
            className,
            paramName);
      // mark param as read so that presentAndNotBeenRead() doesn't trip up
      getParams()->value(getName_c(), paramName);
   }
}

} // namespace PV
