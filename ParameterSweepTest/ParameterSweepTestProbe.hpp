/*
 * ParameterSweepTestProbe.hpp
 *
 *  Created on: Aug 13, 2012
 *      Author: pschultz
 */

#ifndef PARAMETERSWEEPTESTPROBE_HPP_
#define PARAMETERSWEEPTESTPROBE_HPP_

#include "../PetaVision/src/io/StatsProbe.hpp"
#include "../PetaVision/src/layers/HyPerLayer.hpp"
#include <assert.h>
#include <math.h>

namespace PV {

class ParameterSweepTestProbe : StatsProbe {
public:
	ParameterSweepTestProbe(const char * filename, HyPerLayer * layer, const char * msg);
	virtual ~ParameterSweepTestProbe();

	virtual int outputState(double timed);
protected:
    int initParameterSweepTestProbe(const char * filename, HyPerLayer * layer, const char * msg);

private:
    double expectedSum;
    float expectedMin, expectedMax;
};

} /* namespace PV */
#endif /* PARAMETERSWEEPTESTPROBE_HPP_ */
