/*
 * GPUTestForNinesProbe.hpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#ifndef GPUTestForNinesProbe_HPP_
#define GPUTestForNinesProbe_HPP_

#include "../PetaVision/src/io/StatsProbe.hpp"

namespace PV {

class GPUTestForNinesProbe: public PV::StatsProbe {
public:
	GPUTestForNinesProbe(const char * filename, HyPerLayer * layer, const char * msg);
	GPUTestForNinesProbe(HyPerLayer * layer, const char * msg);
	virtual ~GPUTestForNinesProbe();

	virtual int outputState(double timed);

};

} /* namespace PV */
#endif /* GPUTestForNinesProbe_HPP_ */
