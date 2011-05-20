/*
 * LogLatWTAProbe.hpp
 *
 * A derived class of LayerFunctionProbe that uses LogLatWTAFunction
 *
 *  Created on: Apr 26, 2011
 *      Author: peteschultz
 */

#ifndef LOGLATWTAPROBE_HPP_
#define LOGLATWTAPROBE_HPP_

#include "LayerFunctionProbe.hpp"
#include "LogLatWTAFunction.hpp"

namespace PV {

class LogLatWTAProbe : public LayerFunctionProbe {
public:
	LogLatWTAProbe(const char * msg);
	LogLatWTAProbe(const char * filename, HyPerCol * hc, const char * msg);
	~LogLatWTAProbe();
    virtual int outputState(float time, HyPerLayer * l);
};

}  // end namespace PV



#endif /* LOGLATWTAPROBE_HPP_ */
