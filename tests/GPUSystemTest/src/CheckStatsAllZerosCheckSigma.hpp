#ifndef CHECKSTATSALLZEROSCHECKSIGMA_HPP_
#define CHECKSTATSALLZEROSCHECKSIGMA_HPP_

#include "io/PVParams.hpp"
#include "probes/CheckStatsAllZeros.hpp"
#include "probes/ProbeData.hpp"
#include "probes/StatsProbeTypes.hpp"
#include <map>

namespace PV {

class CheckStatsAllZerosCheckSigma : public CheckStatsAllZeros {
  public:
   CheckStatsAllZerosCheckSigma(char const *objName, PVParams *params);
   virtual ~CheckStatsAllZerosCheckSigma();

   virtual std::map<int, LayerStats const>
   checkStats(ProbeData<LayerStats> const &statsBatch) override;

}; // class CheckStatsAllZerosCheckSigma

} // namespace PV

#endif // CHECKSTATSALLZEROSCHECKSIGMA_HPP_
