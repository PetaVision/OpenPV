#include "L0NormLCAProbeLocal.hpp"

namespace PV {

L0NormLCAProbeLocal::L0NormLCAProbeLocal(
      char const *objName,
      PVParams *params) {
   initialize(objName, params);
}

void L0NormLCAProbeLocal::initialize(char const *objName, PVParams *params) {
   L0NormProbeLocal::initialize(objName, params);
}

void L0NormLCAProbeLocal::ioParam_nnzThreshold(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      warnUnnecessaryParameter("VThresh");
   }
}

void L0NormLCAProbeLocal::setNnzThreshold(double nnzThreshold) {
   mNnzThreshold = nnzThreshold;
}

void L0NormLCAProbeLocal::warnUnnecessaryParameter(char const *paramName) {
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
