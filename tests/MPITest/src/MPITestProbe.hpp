/*
 * MPITestProbe.hpp
 *
 *  Created on: Mar 10, 2009
 *      Author: garkenyon
 */

#ifndef MPITESTPROBE_HPP_
#define MPITESTPROBE_HPP_

#include <columns/Communicator.hpp>
#include <params/PVParams.hpp>
#include <probes/StatsProbeImmediate.hpp>

namespace PV {

class MPITestProbe : public PV::StatsProbeImmediate {
  public:
   MPITestProbe(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

  protected:
   virtual void checkStats() override;
   virtual void createProbeLocal(std::shared_ptr<ParamsIO> paramsIO) override;
   virtual void
   createProbeOutputter(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm) override;
   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
}; // end class MPITestProbe

} // end namespace PV

#endif /* MPITESTPROBE_HPP_ */
