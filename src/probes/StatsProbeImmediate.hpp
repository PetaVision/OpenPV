/*
 * StatsProbeImmediate.hpp
 *
 * This class is derived from StatsProbe; the only difference is that immediateMPIAssembly
 * parameter is not used, and the probe always does the MPI assembly as soon as outputState
 * is called. It is an error for a StatsProbeImmediate probe to have immediateMPIAssembly
 * set to false.
 *
 * The motivating use case is for testing, to make it easier to test that the required condition
 * is true on every timestep.
 */

#ifndef STATSPROBEIMMEDIATE_HPP_
#define STATSPROBEIMMEDIATE_HPP_

#include "probes/StatsProbe.hpp"

#include "columns/Communicator.hpp"
#include "io/PVParams.hpp"

#include <memory>

namespace PV {

class StatsProbeImmediate : public StatsProbe {
  protected:
   /**
    * @brief immediateMPIAssembly: The StatsProbeImmediate class does not read the
    * ImmediateMPIAssembly param. It always sets the ImmediateMPIAssembly flag to true.
    */
   virtual void ioParam_immediateMPIAssembly(enum ParamsIOFlag ioFlag) override;

  public:
   StatsProbeImmediate(const char *name, PVParams *params, Communicator const *comm);
   virtual ~StatsProbeImmediate();

  protected:
   StatsProbeImmediate();
   void initialize(const char *name, PVParams *params, Communicator const *comm);
}; // end class StatsProbeImmediate

} /* namespace PV */
#endif /* STATSPROBEIMMEDIATE_HPP_ */
