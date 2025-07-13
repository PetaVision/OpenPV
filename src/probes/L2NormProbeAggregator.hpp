#ifndef L2NORMPROBEAGGREGATOR_HPP_
#define L2NORMPROBEAGGREGATOR_HPP_

#include "probes/NormProbeAggregator.hpp"
#include "probes/ProbeData.hpp"
#include "probes/ProbeDataBuffer.hpp"
#include "structures/MPIBlock.hpp"
#include <memory>

namespace PV {

class L2NormProbeAggregator : public NormProbeAggregator {
  protected:
   virtual void ioParam_exponent(ParamsIOSwitch ioSwitch);

  public:
   L2NormProbeAggregator(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         std::shared_ptr<MPIBlock const> mpiBlock);
   virtual ~L2NormProbeAggregator() {}

   virtual void ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;

  protected:
   L2NormProbeAggregator() {}
   virtual void aggregateNormsBatch(
         ProbeData<double> &aggregatedNormsBatch,
         ProbeData<double> const &partialNormsBatch) override;
   void initialize(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults, std::shared_ptr<MPIBlock const> mpiBlock);

  private:
   double mExponent = 1.0;
   std::shared_ptr<MPIBlock const> mMPIBlock;
   ProbeDataBuffer<double> mStoredValues;

}; // class L2NormProbeAggregator

} // namespace PV

#endif // L2NORMPROBEAGGREGATOR_HPP_
