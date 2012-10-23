/*
 * GPUTestForTwosProbe.hpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#ifndef GPUTestForTwosProbe_HPP_
#define GPUTestForTwosProbe_HPP_

#include "../PetaVision/src/io/StatsProbe.hpp"

namespace PV {

class GPUTestForTwosProbe: public PV::StatsProbe {
public:
	GPUTestForTwosProbe(const char * filename, HyPerLayer * layer, const char * msg);
	GPUTestForTwosProbe(HyPerLayer * layer, const char * msg);
	virtual ~GPUTestForTwosProbe();

	virtual int outputState(double timed);

};

} /* namespace PV */
#endif /* GPUTestForTwosProbe_HPP_ */
