/*
 * ParameterSweepTestProbe.hpp
 *
 *  Created on: Aug 13, 2012
 *      Author: pschultz
 */

#ifndef PARAMETERSWEEPTESTPROBE_HPP_
#define PARAMETERSWEEPTESTPROBE_HPP_

#include "layers/HyPerLayer.hpp"
#include "probes/StatsProbe.hpp"
#include "utils/PVLog.hpp"
#include <cmath>

namespace PV {

class ParameterSweepTestProbe : public StatsProbe {
  public:
   ParameterSweepTestProbe(const char *name, HyPerCol *hc);
   virtual ~ParameterSweepTestProbe();

   virtual Response::Status outputState(double timestamp) override;

  protected:
   int initialize(const char *name, HyPerCol *hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_expectedSum(enum ParamsIOFlag ioFlag);
   virtual void ioParam_expectedMin(enum ParamsIOFlag ioFlag);
   virtual void ioParam_expectedMax(enum ParamsIOFlag ioFlag);

  private:
   double expectedSum;
   float expectedMin, expectedMax;
}; // end class ParameterSweepTestProbe

} // end namespace PV
#endif /* PARAMETERSWEEPTESTPROBE_HPP_ */
