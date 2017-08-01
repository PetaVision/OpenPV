/*
 * StochasticReleaseTestProbe.hpp
 *
 *  Created on: Aug 28, 2013
 *      Author: pschultz
 */

#ifndef STOCHASTICRELEASETESTPROBE_HPP_
#define STOCHASTICRELEASETESTPROBE_HPP_

#include "columns/HyPerCol.hpp"
#include "columns/buildandrun.hpp"
#include "probes/StatsProbe.hpp"
#include <cmath>
#include <stdlib.h>

namespace PV {

class StochasticReleaseTestProbe : public PV::StatsProbe {
  public:
   StochasticReleaseTestProbe(const char *name, HyPerCol *hc);
   virtual ~StochasticReleaseTestProbe();

   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual int outputState(double timed) override;

  protected:
   StochasticReleaseTestProbe();
   int initialize(const char *name, HyPerCol *hc);
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag) override;
   void computePValues();
   void computePValues(int step, int f);

  private:
   int initialize_base();

   // Member variables
  protected:
   HyPerConn *conn = nullptr; // The connection for which targetLayer is the postsynaptic layer.
   // There must be exactly one such conn.
   std::vector<double> pvalues; // The two-tailed p-value of the nnz value of each timestep.
}; // end class StochasticReleaseTestProbe

} /* namespace PV */
#endif /* STOCHASTICRELEASETESTPROBE_HPP_ */
