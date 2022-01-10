/*
 * ArborTestForOnesProbe.hpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#ifndef ARBORTESTFORONESPROBE_HPP_
#define ARBORTESTFORONESPROBE_HPP_

#include "probes/StatsProbe.hpp"

namespace PV {

class ArborTestForOnesProbe : public PV::StatsProbe {
  public:
   ArborTestForOnesProbe(const char *name, PVParams *params, Communicator const *comm);
   virtual ~ArborTestForOnesProbe();

   virtual Response::Status outputState(double simTime, double deltaTime) override;

  protected:
   void initialize(const char *name, PVParams *params, Communicator const *comm);

  private:
   int initialize_base();
};

} /* namespace PV */
#endif // ARBORTESTFORONESPROBE_HPP_
