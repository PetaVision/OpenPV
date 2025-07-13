/*
 * ParameterSweepTestProbe.hpp
 *
 *  Created on: Aug 13, 2012
 *      Author: pschultz
 */

#ifndef PARAMETERSWEEPTESTPROBE_HPP_
#define PARAMETERSWEEPTESTPROBE_HPP_

#include "columns/Communicator.hpp"
#include "params/PVParams.hpp"
#include "probes/StatsProbeImmediate.hpp"

namespace PV {

class ParameterSweepTestProbe : public StatsProbeImmediate {
  protected:
   virtual void ioParam_expectedSum(ParamsIOSwitch ioSwitch);
   virtual void ioParam_expectedMin(ParamsIOSwitch ioSwitch);
   virtual void ioParam_expectedMax(ParamsIOSwitch ioSwitch);

  public:
   ParameterSweepTestProbe(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual ~ParameterSweepTestProbe();

  protected:
   virtual void checkStats() override;
   virtual void createProbeLocal(
        std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults) override;
   void initialize(
         std::shared_ptr<ParamGroup> params,
         std::shared_ptr<ParamGroup> defaults,
         Communicator const *comm);
   virtual int ioParamsFillGroup(ParamsIOSwitch ioSwitch) override;

  private:
   double mExpectedSum = 0.0;
   float mExpectedMin  = 0.0f;
   float mExpectedMax  = 0.0f;
}; // end class ParameterSweepTestProbe

} // end namespace PV
#endif /* PARAMETERSWEEPTESTPROBE_HPP_ */
