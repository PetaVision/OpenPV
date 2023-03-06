#include "MPITestProbeOutputter.hpp"
#include <columns/Communicator.hpp>
#include <include/pv_common.h>
#include <io/PVParams.hpp>
#include <io/PrintStream.hpp>
#include <probes/StatsProbeOutputter.hpp>
#include <probes/StatsProbeTypes.hpp>
#include <utils/PVLog.hpp>

#include <cmath>
#include <memory>

namespace PV {

MPITestProbeOutputter::MPITestProbeOutputter(
      char const *objName,
      PVParams *params,
      Communicator const *comm)
      : StatsProbeOutputter(objName, params, comm) {}

MPITestProbeOutputter::~MPITestProbeOutputter() {}

void MPITestProbeOutputter::printGlobalXPosStats(
      ProbeData<LayerStats> const &stats,
      float min_global_xpos,
      float max_global_xpos,
      double ave_global_xpos) {
   float const tol = 1.0e-4f;

   double simTime = stats.getTimestamp();
   int status     = PV_SUCCESS;
   int nbatch     = static_cast<int>(stats.size());
   for (int b = 0; b < nbatch; b++) {
      auto outputStream = returnOutputStream(b);
      if (!outputStream) {
         continue;
      }
      outputStream->printf(
            "%s min_global_xpos==%f ave_global_xpos==%f max_global_xpos==%f\n",
            getMessage().c_str(),
            (double)min_global_xpos,
            (double)ave_global_xpos,
            (double)max_global_xpos);

      LayerStats statsElem = stats.getValue(b);
      if (std::fabs(statsElem.mMin / min_global_xpos - 1.0f) >= tol) {
         ErrorLog().printf(
               "Probe %s, t=%f, batch index %d, min %f differs from correct value %f\n",
               getName().c_str(),
               simTime,
               b,
               (double)statsElem.mMin,
               (double)min_global_xpos);
         status = PV_FAILURE;
      }
      if (std::fabs(statsElem.mMax / max_global_xpos - 1.0f) >= tol) {
         ErrorLog().printf(
               "Probe %s, t=%f, batch index %d, max %f differs from correct value %f\n",
               getName().c_str(),
               simTime,
               b,
               (double)statsElem.mMax,
               (double)max_global_xpos);
         status = PV_FAILURE;
      }
      if (std::fabs(statsElem.average() / ave_global_xpos - 1.0) >= (double)tol) {
         ErrorLog().printf(
               "Probe %s, t=%f, batch index %d, average %f differs from correct value %f\n",
               getName().c_str(),
               simTime,
               b,
               statsElem.average(),
               ave_global_xpos);
         status = PV_FAILURE;
      }
   }
   FatalIf(status != PV_SUCCESS, "Test failed.\n");
}

} // namespace PV
