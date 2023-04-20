/*
 * StochasticReleaseTestProbe.hpp
 *
 *  Created on: Aug 28, 2013
 *      Author: pschultz
 */

#ifndef STOCHASTICRELEASETESTPROBE_HPP_
#define STOCHASTICRELEASETESTPROBE_HPP_

#include "columns/Communicator.hpp"
#include "columns/ComponentBasedObject.hpp"
#include "columns/Messages.hpp"
#include "io/PVParams.hpp"
#include "observerpattern/Response.hpp"
#include "probes/StatsProbeImmediate.hpp"

#include <memory>
#include <vector>

namespace PV {

class StochasticReleaseTestProbe : public PV::StatsProbeImmediate {
  public:
   StochasticReleaseTestProbe(const char *name, PVParams *params, Communicator const *comm);
   virtual ~StochasticReleaseTestProbe();

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

  protected:
   StochasticReleaseTestProbe();
   virtual void checkStats() override;
   virtual void createProbeLocal(char const *name, PVParams *params) override;
   virtual void
   createProbeOutputter(char const *name, PVParams *params, Communicator const *comm) override;
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   int computePValues();

   // Member variables
  protected:
   ComponentBasedObject *mConn = nullptr; // The connection for which targetLayer is the post layer.
   // There must be exactly one such conn.
   std::vector<double> m_pValues; // The two-tailed p-value of the nnz value of each timestep.
}; // end class StochasticReleaseTestProbe

} /* namespace PV */
#endif /* STOCHASTICRELEASETESTPROBE_HPP_ */
