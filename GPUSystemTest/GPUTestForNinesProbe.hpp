/*
 * GPUTestForNinesProbe.hpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#ifndef GPUTestForNinesProbe_HPP_
#define GPUTestForNinesProbe_HPP_

#include <io/StatsProbe.hpp>

namespace PV {

class GPUTestForNinesProbe: public PV::StatsProbe {
public:
	GPUTestForNinesProbe(const char * probeName, HyPerCol * hc);
	virtual ~GPUTestForNinesProbe();

	virtual int outputState(double timed);

protected:
    int initGPUTestForNinesProbe(const char * probeName, HyPerCol * hc);

private:
	int initGPUTestForNinesProbe_base();

};

} /* namespace PV */
#endif /* GPUTestForNinesProbe_HPP_ */
