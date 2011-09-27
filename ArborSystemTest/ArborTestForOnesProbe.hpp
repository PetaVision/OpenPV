/*
 * ArborTestForOnesProbe.hpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#ifndef ArborTestForOnesProbe_HPP_
#define ArborTestForOnesProbe_HPP_

#include "../PetaVision/src/io/StatsProbe.hpp"

namespace PV {

class ArborTestForOnesProbe: public PV::StatsProbe {
public:
	ArborTestForOnesProbe(const char * filename, HyPerCol * hc, const char * msg);
	ArborTestForOnesProbe(const char * msg);

	virtual int outputState(float time, HyPerLayer * l	);

};

} /* namespace PV */
#endif /* ArborTestForOnesProbe_HPP_ */
