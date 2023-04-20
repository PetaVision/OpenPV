#include "StochasticReleaseTestProbeOutputter.hpp"
#include <columns/Communicator.hpp>
#include <io/PVParams.hpp>
#include <probes/StatsProbeOutputter.hpp>

namespace PV {

StochasticReleaseTestProbeOutputter::StochasticReleaseTestProbeOutputter(
      char const *objName,
      PVParams *params,
      Communicator const *comm) {
   initialize(objName, params, comm);
}

StochasticReleaseTestProbeOutputter::StochasticReleaseTestProbeOutputter() {}

StochasticReleaseTestProbeOutputter::~StochasticReleaseTestProbeOutputter() {}

void StochasticReleaseTestProbeOutputter::initialize(
      char const *objName,
      PVParams *params,
      Communicator const *comm) {
   StatsProbeOutputter::initialize(objName, params, comm);
}

void StochasticReleaseTestProbeOutputter::printNumNonzeroData(
      int f,
      int nnzf,
      double mean,
      double stddev,
      double numdevs,
      double pval) {
   auto outputStream = returnOutputStream(0);
   if (outputStream) {
      outputStream->printf(
            "    Feature %d, nnz=%5d, expectation=%7.1f, std.dev.=%5.1f, discrepancy of %f "
            "deviations, p-value %f\n",
            f,
            nnzf,
            mean,
            stddev,
            numdevs,
            pval);
   }
}

} // namespace PV
