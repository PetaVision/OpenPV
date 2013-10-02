/*
 * DelayTestProbe.hpp
 *
 *  Created on: October 1, 2013
 *      Author: wchavez
 */

#ifndef DelayTestProbe_HPP_
#define DelayTestProbe_HPP_

#include <io/StatsProbe.hpp>

namespace PV {

class DelayTestProbe: public PV::StatsProbe {
public:
   DelayTestProbe(const char * filename, HyPerLayer * layer, const char * msg);
   DelayTestProbe(HyPerLayer * layer, const char * msg);
	virtual ~DelayTestProbe();

	virtual int outputState(double timed);

};

} /* namespace PV */
#endif /* DelayTestProbe_HPP_ */
