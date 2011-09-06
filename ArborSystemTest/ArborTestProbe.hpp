/*
 * ArborTestProbe.hpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#ifndef ArborTestProbe_HPP_
#define ArborTestProbe_HPP_

#include "../PetaVision/src/io/StatsProbe.hpp"

namespace PV {

class ArborTestProbe: public PV::StatsProbe {
public:
	ArborTestProbe(const char * filename, HyPerCol * hc, const char * msg);
	ArborTestProbe(const char * msg);

	virtual int outputState(float time, HyPerLayer * l	);

};

} /* namespace PV */
#endif /* ArborTestProbe_HPP_ */
