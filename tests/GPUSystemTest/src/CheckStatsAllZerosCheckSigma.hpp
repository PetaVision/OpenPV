#ifndef CHECKSTATSALLZEROSCHECKSIGMA_HPP_
#define CHECKSTATSALLZEROSCHECKSIGMA_HPP_

#include "params/ParamGroup.hpp"
#include "probes/CheckStatsAllZeros.hpp"
#include "probes/ProbeData.hpp"
#include "probes/StatsProbeTypes.hpp"
#include <map>

namespace PV {

class CheckStatsAllZerosCheckSigma : public CheckStatsAllZeros {
  public:
   CheckStatsAllZerosCheckSigma(
         std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults);
   virtual ~CheckStatsAllZerosCheckSigma();

   virtual std::map<int, LayerStats const>
   checkStats(ProbeData<LayerStats> const &statsBatch) override;

}; // class CheckStatsAllZerosCheckSigma

} // namespace PV

#endif // CHECKSTATSALLZEROSCHECKSIGMA_HPP_
