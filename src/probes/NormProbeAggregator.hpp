#ifndef NORMPROBEAGGREGATOR_HPP_
#define NORMPROBEAGGREGATOR_HPP_

#include "io/PVParams.hpp"
#include "probes/ProbeComponent.hpp"
#include "probes/ProbeData.hpp"
#include "probes/ProbeDataBuffer.hpp"
#include "structures/MPIBlock.hpp"
#include <memory>

namespace PV {

class NormProbeAggregator : public ProbeComponent {
  public:
   NormProbeAggregator(
         char const *objName,
         PVParams *params,
         std::shared_ptr<MPIBlock const> mpiBlock);
   virtual ~NormProbeAggregator() {}

   void aggregateStoredValues(ProbeDataBuffer<double> const &partialStore);
   void clearStoredValues();
   virtual void ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   ProbeDataBuffer<double> const &getStoredValues() const { return mStoredValues; }

  protected:
   NormProbeAggregator() {}
   virtual void aggregateNormsBatch(
         ProbeData<double> &aggregatedNormsBatch,
         ProbeData<double> const &partialNormsBatch);
   void initialize(char const *objName, PVParams *params, std::shared_ptr<MPIBlock const> mpiBlock);

  private:
   std::shared_ptr<MPIBlock const> mMPIBlock;
   ProbeDataBuffer<double> mStoredValues;

}; // class NormProbeAggregator

} // namespace PV

#endif // L1NORMPROBEAGGREGATOR_HPP_
