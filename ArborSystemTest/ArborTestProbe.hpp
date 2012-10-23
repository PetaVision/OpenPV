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
	ArborTestProbe(const char * filename, HyPerLayer * layer, const char * msg);
	ArborTestProbe(HyPerLayer * layer, const char * msg);
	virtual ~ArborTestProbe();

	virtual int outputState(double timed);

};

} /* namespace PV */
#endif /* ArborTestProbe_HPP_ */
