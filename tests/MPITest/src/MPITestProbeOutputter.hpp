#ifndef MPITESTPROBEOUTPUTTER_HPP_
#define MPITESTPROBEOUTPUTTER_HPP_

#include <columns/Communicator.hpp>
#include <params/PVParams.hpp>
#include <probes/ProbeData.hpp>
#include <probes/StatsProbeOutputter.hpp>
#include <probes/StatsProbeTypes.hpp>

namespace PV {

class MPITestProbeOutputter : public StatsProbeOutputter {
  public:
   MPITestProbeOutputter(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ~MPITestProbeOutputter();

   void printGlobalXPosStats(
         ProbeData<LayerStats> const &stats,
         float min_global_xpos,
         float max_global_xpos,
         double ave_global_xpos);
};

} // namespace PV

#endif // MPITESTPROBEOUTPUTTER_HPP_
