/*
 * InitWeightTestProbe.hpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#ifndef InitWeightTestProbe_HPP_
#define InitWeightTestProbe_HPP_

#include <io/StatsProbe.hpp>

namespace PV {

class InitWeightTestProbe: public PV::StatsProbe {
public:
	InitWeightTestProbe(const char * probeName, HyPerCol * hc);

	virtual int outputState(double timef);

protected:
	int initInitWeightTestProbe(const char * probeName, HyPerCol * hc);
	virtual void ioParam_buffer(enum ParamsIOFlag ioFlag);

private:
	int initInitWeightTestProbe_base();
};

BaseObject * createInitWeightTestProbe(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif /* ArborTestProbe_HPP_ */
