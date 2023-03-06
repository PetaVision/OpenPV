#ifndef STOCHASTICRELEASETESTPROBEOUTPUTTER_HPP_
#define STOCHASTICRELEASETESTPROBEOUTPUTTER_HPP_

#include <columns/Communicator.hpp>
#include <io/PVParams.hpp>
#include <probes/StatsProbeOutputter.hpp>

namespace PV {

class StochasticReleaseTestProbeOutputter : public StatsProbeOutputter {
  public:
   StochasticReleaseTestProbeOutputter(
         char const *objName,
         PVParams *params,
         Communicator const *comm);
   virtual ~StochasticReleaseTestProbeOutputter();

   void
   printNumNonzeroData(int f, int nnzf, double mean, double stddev, double numdevs, double pval);

  protected:
   StochasticReleaseTestProbeOutputter();
   void initialize(char const *objName, PVParams *params, Communicator const *comm);
};

} // namespace PV

#endif // STOCHASTICRELEASETESTPROBEOUTPUTTER_HPP_
