/*
 * ArborTestProbe.hpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#ifndef ARBORTESTPROBE_HPP_
#define ARBORTESTPROBE_HPP_

#include "columns/Communicator.hpp"
#include "io/PVParams.hpp"
#include "probes/StatsProbeImmediate.hpp"

namespace PV {

class ArborTestProbe : public PV::StatsProbeImmediate {
  public:
   ArborTestProbe(const char *name, PVParams *params, Communicator const *comm);
   virtual ~ArborTestProbe();

  protected:
   virtual void checkStats() override;
   virtual void createProbeLocal(char const *name, PVParams *params) override;
   void initialize(const char *name, PVParams *params, Communicator const *comm);
};

} /* namespace PV */
#endif // ARBORTESTPROBE_HPP_
