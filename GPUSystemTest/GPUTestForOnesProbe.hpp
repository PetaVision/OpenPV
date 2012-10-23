/*
 * GPUTestForOnesProbe.hpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#ifndef GPUTestForOnesProbe_HPP_
#define GPUTestForOnesProbe_HPP_

#include "../PetaVision/src/io/StatsProbe.hpp"

namespace PV {

class GPUTestForOnesProbe: public PV::StatsProbe {
public:
	GPUTestForOnesProbe(const char * filename, HyPerLayer * layer, const char * msg);
	GPUTestForOnesProbe(HyPerLayer * layer, const char * msg);
	virtual ~GPUTestForOnesProbe();

	virtual int outputState(double timed);

};

} /* namespace PV */
#endif /* GPUTestForOnesProbe_HPP_ */
