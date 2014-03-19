/*
 * GPUTestForOnesProbe.hpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#ifndef GPUTestForOnesProbe_HPP_
#define GPUTestForOnesProbe_HPP_

#include <io/StatsProbe.hpp>

namespace PV {

class GPUTestForOnesProbe: public PV::StatsProbe {
public:
	GPUTestForOnesProbe(const char * probeName, HyPerCol * hc);
	virtual ~GPUTestForOnesProbe();

	virtual int outputState(double timed);

protected:
    int initGPUTestForOnesProbe(const char * probeName, HyPerCol * hc);

private:
    int initGPUTestForOnesProbe_base();

};

} /* namespace PV */
#endif /* GPUTestForOnesProbe_HPP_ */
