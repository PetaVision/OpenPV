/*
 * ParameterSweepTestProbe.hpp
 *
 *  Created on: Aug 13, 2012
 *      Author: pschultz
 */

#ifndef PARAMETERSWEEPTESTPROBE_HPP_
#define PARAMETERSWEEPTESTPROBE_HPP_

#include <io/StatsProbe.hpp>
#include <layers/HyPerLayer.hpp>
#include <assert.h>
#include <math.h>

namespace PV {

class ParameterSweepTestProbe : public StatsProbe {
public:
	ParameterSweepTestProbe(const char * probeName, HyPerCol * hc);
	virtual ~ParameterSweepTestProbe();

	virtual int outputState(double timed);
protected:
    int initParameterSweepTestProbe(const char * probeName, HyPerCol * hc);
    virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
    virtual void ioParam_buffer(enum ParamsIOFlag ioFlag);
    virtual void ioParam_expectedSum(enum ParamsIOFlag ioFlag);
    virtual void ioParam_expectedMin(enum ParamsIOFlag ioFlag);
    virtual void ioParam_expectedMax(enum ParamsIOFlag ioFlag);

private:
    double expectedSum;
    float expectedMin, expectedMax;
}; // end class ParameterSweepTestProbe

BaseObject * createParameterSweepTestProbe(char const * name, HyPerCol * hc);

} // end namespace PV
#endif /* PARAMETERSWEEPTESTPROBE_HPP_ */
