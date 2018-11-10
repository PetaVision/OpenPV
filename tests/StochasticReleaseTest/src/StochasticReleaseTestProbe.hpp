/*
 * StochasticReleaseTestProbe.hpp
 *
 *  Created on: Aug 28, 2013
 *      Author: pschultz
 */

#ifndef STOCHASTICRELEASETESTPROBE_HPP_
#define STOCHASTICRELEASETESTPROBE_HPP_

#include "columns/ComponentBasedObject.hpp"
#include "columns/buildandrun.hpp"
#include "probes/StatsProbe.hpp"
#include <cmath>
#include <stdlib.h>

namespace PV {


class StochasticReleaseTestProbe : public PV::StatsProbe {
  public:
   StochasticReleaseTestProbe(const char *name, PVParams *params, Communicator *comm);
   virtual ~StochasticReleaseTestProbe();

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status outputState(double simTime, double deltaTime) override;

  protected:
   StochasticReleaseTestProbe();
   void initialize(const char *name, PVParams *params, Communicator *comm);
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag) override;
   void computePValues();
   void computePValues(int step, int f);

  private:
   int initialize_base();

   // Member variables
  protected:
   ComponentBasedObject *conn = nullptr; // The connection for which targetLayer is the post layer.
   // There must be exactly one such conn.
   std::vector<double> pvalues; // The two-tailed p-value of the nnz value of each timestep.
}; // end class StochasticReleaseTestProbe

} /* namespace PV */
#endif /* STOCHASTICRELEASETESTPROBE_HPP_ */
