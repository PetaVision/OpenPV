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
  public:
   MPITestProbe(const char *name, PVParams *params, Communicator *comm);

   virtual Response::Status outputState(double simTime, double deltaTime) override;

  protected:
   void initialize(const char *name, PVParams *params, Communicator *comm);

  private:
   int initialize_base();
}; // end class MPITestProbe

} // end namespace PV

#endif /* MPITESTPROBE_HPP_ */
