/*
 * GPUTestProbe.hpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#ifndef GPUTestProbe_HPP_
#define GPUTestProbe_HPP_

#include "../PetaVision/src/io/StatsProbe.hpp"

namespace PV {

class GPUTestProbe: public PV::StatsProbe {
public:
	GPUTestProbe(const char * filename, HyPerLayer * layer, const char * msg);
	GPUTestProbe(HyPerLayer * layer, const char * msg);
	virtual ~GPUTestProbe();

	virtual int outputState(float timef);

};

} /* namespace PV */
#endif /* GPUTestProbe_HPP_ */
