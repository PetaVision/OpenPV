#include "L2NormProbeAggregator.hpp"
#include "arch/mpi/mpi.h"
#include "cMakeHeader.h"
#include "utils/PVAssert.hpp"
#include <cmath>

namespace PV {

L2NormProbeAggregator::L2NormProbeAggregator(std::shared_ptr<ParamsIO> paramsIO, std::shared_ptr<MPIBlock const> mpiBlock) {
   initialize(paramsIO, mpiBlock);
}

void L2NormProbeAggregator::aggregateNormsBatch(
      ProbeData<double> &aggregatedNormsBatch,
      ProbeData<double> const &partialNormsBatch) {
   NormProbeAggregator::aggregateNormsBatch(aggregatedNormsBatch, partialNormsBatch);

   if (mExponent != 2.0) {
      // Raise each value by the power (mExponent / 2.0)
      double power = mExponent / 2.0;
      int nbatch   = static_cast<int>(partialNormsBatch.size());
      for (int b = 0; b < nbatch; ++b) {
         double &x = aggregatedNormsBatch.getValue(b);
         x         = std::pow(x, power);
      }
   }
}

void L2NormProbeAggregator::initialize(std::shared_ptr<ParamsIO> paramsIO, std::shared_ptr<MPIBlock const> mpiBlock) {
   NormProbeAggregator::initialize(paramsIO, mpiBlock);
}

void L2NormProbeAggregator::ioParam_exponent(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "exponent", &mExponent);
}

void L2NormProbeAggregator::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   ioParam_exponent(ioSwitch);
}

} // namespace PV
