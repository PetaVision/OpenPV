/*
 * GPUTestProbe.hpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#ifndef GPUTestProbe_HPP_
#define GPUTestProbe_HPP_

#include <io/StatsProbe.hpp>

namespace PV {

class GPUTestProbe: public PV::StatsProbe {
public:
	GPUTestProbe(const char * filename, HyPerLayer * layer, const char * msg);
	GPUTestProbe(HyPerLayer * layer, const char * msg);
	virtual ~GPUTestProbe();

	virtual int outputState(double timed);

};

} /* namespace PV */
#endif /* GPUTestProbe_HPP_ */
