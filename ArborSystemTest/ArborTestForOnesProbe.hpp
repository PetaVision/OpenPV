/*
 * ArborTestForOnesProbe.hpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#ifndef ArborTestForOnesProbe_HPP_
#define ArborTestForOnesProbe_HPP_

#include <io/StatsProbe.hpp>

namespace PV {

class ArborTestForOnesProbe: public PV::StatsProbe {
public:
	ArborTestForOnesProbe(const char * filename, HyPerLayer * layer, const char * msg);
	ArborTestForOnesProbe(HyPerLayer * layer, const char * msg);
	virtual ~ArborTestForOnesProbe();

	virtual int outputState(double timed);

};

} /* namespace PV */
#endif /* ArborTestForOnesProbe_HPP_ */
