/*
 * InitWeightTestProbe.hpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#ifndef InitWeightTestProbe_HPP_
#define InitWeightTestProbe_HPP_

#include "../PetaVision/src/io/StatsProbe.hpp"

namespace PV {

class InitWeightTestProbe: public PV::StatsProbe {
public:
	InitWeightTestProbe(const char * filename, HyPerLayer * layer, const char * msg);
	InitWeightTestProbe(HyPerLayer * layer, const char * msg);

	virtual int outputState(double timef);

};

} /* namespace PV */
#endif /* ArborTestProbe_HPP_ */
