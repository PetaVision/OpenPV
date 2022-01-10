/*
 * ArborTestProbe.hpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#ifndef ARBORTESTPROBE_HPP_
#define ARBORTESTPROBE_HPP_

#include "probes/StatsProbe.hpp"

namespace PV {

class ArborTestProbe : public PV::StatsProbe {
  public:
   ArborTestProbe(const char *name, PVParams *params, Communicator const *comm);
   virtual ~ArborTestProbe();

   virtual Response::Status outputState(double simTime, double deltaTime) override;

  protected:
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag) override;

  private:
   int initialize_base();
};

} /* namespace PV */
#endif // ARBORTESTPROBE_HPP_
