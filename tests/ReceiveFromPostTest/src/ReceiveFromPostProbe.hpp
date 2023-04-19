/*
 * ReceiveFromPostProbe.hpp
 * Author: slundquist
 */

#ifndef RECEIVEFROMPOSTPROBE_HPP_
#define RECEIVEFROMPOSTPROBE_HPP_

#include <columns/Communicator.hpp>
#include <io/PVParams.hpp>
#include <probes/StatsProbeImmediate.hpp>

namespace PV {

class ReceiveFromPostProbe : public PV::StatsProbeImmediate {
  protected:
   void ioParam_tolerance(enum ParamsIOFlag ioFlag);

  public:
   ReceiveFromPostProbe(const char *name, PVParams *params, Communicator const *comm);

  protected:
   virtual void checkStats() override;
   virtual void createProbeLocal(char const *name, PVParams *params) override;
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

   // Member variables
  protected:
   float mTolerance = 1.0e-3f;

}; // end class ReceiveFromPostProbe

} // namespace PV

#endif // RECEIVEFROMPOSTPROBE_HPP_
