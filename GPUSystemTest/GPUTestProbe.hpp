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
	GPUTestProbe(const char * probeName, HyPerCol * hc);
	virtual ~GPUTestProbe();

	virtual int outputState(double timed);

protected:
	int initGPUTestProbe(const char * probeName, HyPerCol * hc);

private:
    int initGPUTestProbe_base();

};

} /* namespace PV */
#endif /* GPUTestProbe_HPP_ */
