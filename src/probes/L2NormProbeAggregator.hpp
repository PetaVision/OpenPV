#ifndef L2NORMPROBEAGGREGATOR_HPP_
#define L2NORMPROBEAGGREGATOR_HPP_

#include "io/PVParams.hpp"
#include "probes/NormProbeAggregator.hpp"
#include "probes/ProbeData.hpp"
#include "probes/ProbeDataBuffer.hpp"
#include "structures/MPIBlock.hpp"
#include <memory>

namespace PV {

class L2NormProbeAggregator : public NormProbeAggregator {
  protected:
   virtual void ioParam_exponent(enum ParamsIOFlag ioFlag);

  public:
   L2NormProbeAggregator(
         char const *objName,
         PVParams *params,
         std::shared_ptr<MPIBlock const> mpiBlock);
   virtual ~L2NormProbeAggregator() {}

   virtual void ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

  protected:
   L2NormProbeAggregator() {}
   virtual void aggregateNormsBatch(
         ProbeData<double> &aggregatedNormsBatch,
         ProbeData<double> const &partialNormsBatch) override;
   void initialize(char const *objName, PVParams *params, std::shared_ptr<MPIBlock const> mpiBlock);

  private:
   double mExponent = 1.0;
   std::shared_ptr<MPIBlock const> mMPIBlock;
   ProbeDataBuffer<double> mStoredValues;

}; // class L2NormProbeAggregator

} // namespace PV

#endif // L2NORMPROBEAGGREGATOR_HPP_
