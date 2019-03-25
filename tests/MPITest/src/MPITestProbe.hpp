/*
 * MPITestProbe.hpp
 *
 *  Created on: Mar 10, 2009
 *      Author: garkenyon
 */

#ifndef MPITESTPROBE_HPP_
#define MPITESTPROBE_HPP_

#include "probes/StatsProbe.hpp"

namespace PV {

class MPITestProbe : public PV::StatsProbe {
  protected:
   /**
    * MPITestProbe sets buffer to "Activity". It is an error to set it to a different value.
    */
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag) override;

  public:
   MPITestProbe(const char *name, PVParams *params, Communicator const *comm);

   virtual Response::Status outputState(double simTime, double deltaTime) override;

  protected:
   void initialize(const char *name, PVParams *params, Communicator const *comm);
}; // end class MPITestProbe

} // end namespace PV

#endif /* MPITESTPROBE_HPP_ */
