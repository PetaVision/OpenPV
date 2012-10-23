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
   LogLatWTAProbe(HyPerLayer * layer, const char * msg);
   LogLatWTAProbe(const char * filename, HyPerLayer * layer, const char * msg);
   virtual ~LogLatWTAProbe();

protected:
   int initLogLatWTAProbe(const char * filename, HyPerLayer * layer, const char * msg);
   virtual int writeState(double timed, HyPerLayer * l, pvdata_t value);
};

}  // end namespace PV



#endif /* LOGLATWTAPROBE_HPP_ */
