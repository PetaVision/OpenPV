/*
 * MPITestProbe.hpp
 *
 *  Created on: Mar 10, 2009
 *      Author: garkenyon
 */

#ifndef MPITESTPROBE_HPP_
#define MPITESTPROBE_HPP_

#include <columns/Communicator.hpp>
#include <io/PVParams.hpp>
#include <probes/StatsProbeImmediate.hpp>

namespace PV {

class MPITestProbe : public PV::StatsProbeImmediate {
  public:
   MPITestProbe(const char *name, PVParams *params, Communicator const *comm);

  protected:
   virtual void checkStats() override;
   virtual void createProbeLocal(char const *name, PVParams *params) override;
   virtual void
   createProbeOutputter(char const *name, PVParams *params, Communicator const *comm) override;
   void initialize(char const *name, PVParams *params, Communicator const *comm);
}; // end class MPITestProbe

} // end namespace PV

#endif /* MPITESTPROBE_HPP_ */
