#ifndef STATSPROBEAGGREGATOR_HPP_
#define STATSPROBEAGGREGATOR_HPP_

#include "io/PVParams.hpp"
#include "probes/ProbeComponent.hpp"
#include "probes/ProbeDataBuffer.hpp"
#include "probes/StatsProbeTypes.hpp"
#include "structures/MPIBlock.hpp"

#include <memory>

namespace PV {

class StatsProbeAggregator : public ProbeComponent {
  public:
   StatsProbeAggregator(
         char const *objName,
         PVParams *params,
         std::shared_ptr<MPIBlock const> mpiBlock);
   ~StatsProbeAggregator() {}

   void aggregateStoredValues(ProbeDataBuffer<LayerStats> const &partialStore);
   void clearStoredValues();
   void ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   ProbeDataBuffer<LayerStats> const &getStoredValues() const { return mStoredValues; }

  protected:
   StatsProbeAggregator() {}
   void initialize(char const *objName, PVParams *params, std::shared_ptr<MPIBlock const> mpiBlock);

  private:
   void broadcastStoredValues();

  private:
   std::shared_ptr<MPIBlock const> mMPIBlock;
   ProbeDataBuffer<LayerStats> mStoredValues;

}; // class StatsProbeAggregator

} // namespace PV

#endif // STATSPROBEAGGREGATOR_HPP_
