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
	InitWeightTestProbe(const char * filename, HyPerCol * hc, const char * msg);
	InitWeightTestProbe(const char * msg);

	virtual int outputState(float time, HyPerLayer * l	);

};

} /* namespace PV */
#endif /* ArborTestProbe_HPP_ */
