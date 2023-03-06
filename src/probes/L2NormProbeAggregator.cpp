#include "L2NormProbeAggregator.hpp"
#include "arch/mpi/mpi.h"
#include "cMakeHeader.h"
#include "utils/PVAssert.hpp"
#include <cmath>

namespace PV {

L2NormProbeAggregator::L2NormProbeAggregator(
      char const *objName,
      PVParams *params,
      std::shared_ptr<MPIBlock const> mpiBlock) {
   initialize(objName, params, mpiBlock);
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

void L2NormProbeAggregator::initialize(
      char const *objName,
      PVParams *params,
      std::shared_ptr<MPIBlock const> mpiBlock) {
   NormProbeAggregator::initialize(objName, params, mpiBlock);
}

void L2NormProbeAggregator::ioParam_exponent(enum ParamsIOFlag ioFlag) {
   getParams()->ioParamValue(
         ioFlag, getName_c(), "exponent", &mExponent, mExponent, true /*warnIfAbsent*/);
}

void L2NormProbeAggregator::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_exponent(ioFlag);
}

} // namespace PV
