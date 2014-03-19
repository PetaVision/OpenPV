/*
 * GPUTestForTwosProbe.hpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#ifndef GPUTestForTwosProbe_HPP_
#define GPUTestForTwosProbe_HPP_

#include <io/StatsProbe.hpp>

namespace PV {

class GPUTestForTwosProbe: public PV::StatsProbe {
public:
	GPUTestForTwosProbe(const char * probeName, HyPerCol * hc);
	virtual ~GPUTestForTwosProbe();

	virtual int outputState(double timed);

protected:
    int initGPUTestForTwosProbe(const char * probeName, HyPerCol * hc);

private:
    int initGPUTestForTwosProbe_base();

};

} /* namespace PV */
#endif /* GPUTestForTwosProbe_HPP_ */
