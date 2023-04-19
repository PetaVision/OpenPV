/*
 * ArborTestForOnesProbe.hpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#ifndef ARBORTESTFORONESPROBE_HPP_
#define ARBORTESTFORONESPROBE_HPP_

#include "columns/Communicator.hpp"
#include "io/PVParams.hpp"
#include "probes/StatsProbeImmediate.hpp"

namespace PV {

class ArborTestForOnesProbe : public PV::StatsProbeImmediate {
  public:
   ArborTestForOnesProbe(const char *name, PVParams *params, Communicator const *comm);
   virtual ~ArborTestForOnesProbe();

  protected:
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual void checkStats() override;
   int checkValue(float value, double timestamp, int batchIndex, char const *desc);
};

} /* namespace PV */
#endif // ARBORTESTFORONESPROBE_HPP_
