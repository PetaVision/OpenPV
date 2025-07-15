/*
 * ReceiveFromPostProbe.hpp
 * Author: slundquist
 */

#ifndef RECEIVEFROMPOSTPROBE_HPP_
#define RECEIVEFROMPOSTPROBE_HPP_

#include <columns/Communicator.hpp>
#include <params/PVParams.hpp>
#include <probes/StatsProbeImmediate.hpp>

namespace PV {

class ReceiveFromPostProbe : public PV::StatsProbeImmediate {
  protected:
   void ioParam_tolerance(ParamsIOSwitch ioSwitch);

  public:
   ReceiveFromPostProbe(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

  protected:
   virtual void checkStats() override;
   virtual void createProbeLocal(std::shared_ptr<ParamsIO> paramsIO) override;
   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;

   // Member variables
  protected:
   float mTolerance = 1.0e-3f;

}; // end class ReceiveFromPostProbe

} // namespace PV

#endif // RECEIVEFROMPOSTPROBE_HPP_
