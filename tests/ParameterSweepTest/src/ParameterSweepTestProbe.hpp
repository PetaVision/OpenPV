/*
 * ParameterSweepTestProbe.hpp
 *
 *  Created on: Aug 13, 2012
 *      Author: pschultz
 */

#ifndef PARAMETERSWEEPTESTPROBE_HPP_
#define PARAMETERSWEEPTESTPROBE_HPP_

#include "columns/Communicator.hpp"
#include "io/PVParams.hpp"
#include "probes/StatsProbeImmediate.hpp"

namespace PV {

class ParameterSweepTestProbe : public StatsProbeImmediate {
  protected:
   virtual void ioParam_expectedSum(enum ParamsIOFlag ioFlag);
   virtual void ioParam_expectedMin(enum ParamsIOFlag ioFlag);
   virtual void ioParam_expectedMax(enum ParamsIOFlag ioFlag);

  public:
   ParameterSweepTestProbe(const char *name, PVParams *params, Communicator const *comm);
   virtual ~ParameterSweepTestProbe();

  protected:
   virtual void checkStats() override;
   virtual void createProbeLocal(char const *name, PVParams *params) override;
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;

  private:
   double mExpectedSum = 0.0;
   float mExpectedMin  = 0.0f;
   float mExpectedMax  = 0.0f;
}; // end class ParameterSweepTestProbe

} // end namespace PV
#endif /* PARAMETERSWEEPTESTPROBE_HPP_ */
