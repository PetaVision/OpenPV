#include "L0NormLCAProbeLocal.hpp"

namespace PV {

L0NormLCAProbeLocal::L0NormLCAProbeLocal(std::shared_ptr<ParamsIO> paramsIO) {
   initialize(paramsIO);
}

void L0NormLCAProbeLocal::initialize(std::shared_ptr<ParamsIO> paramsIO) {
   L0NormProbeLocal::initialize(paramsIO);
}

void L0NormLCAProbeLocal::ioParam_nnzThreshold(ParamsIOSwitch ioSwitch) {
   if (ioSwitch == ParamsIOSwitch::Read) {
      warnUnnecessaryParameter("nnzThreshold");
   }
}

void L0NormLCAProbeLocal::setNnzThreshold(double nnzThreshold) {
   mNnzThreshold = nnzThreshold;
}

void L0NormLCAProbeLocal::warnUnnecessaryParameter(char const *paramName) {
   if (mParamsIO->isPresent(paramName)) {
      char const *className = mParamsIO->getKeyword().c_str();
      WarnLog().printf(
            "Parameter %s is present in the params file for %s \"%s\", but %s does not use it. "
            "Instead, %s is taken from the target layer.\n",
            paramName,
            className,
            getName_c(),
            className,
            paramName);
      // mark param as read so that presentAndNotBeenRead() doesn't trip up
      double paramValue = mParamsIO->readValue<double>(paramName, false /*warnIfAbsentFlag*/);
   }
}

} // namespace PV
