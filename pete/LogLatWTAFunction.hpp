/*
 * LogLatWTAFunction.hpp
 *
 * Computes sum( log(1+a'*L*a) ) for a layer
 * The sum is over the x- and y-directions
 * a'*L*a is the lateral winner-take-all term over features at one (x,y)
 *
 *  Created on: Apr 26, 2011
 *      Author: peteschultz
 */

#ifndef LOGLATWTAFUNCTION_HPP_
#define LOGLATWTAFUNCTION_HPP_

#include "LayerFunction.hpp"

namespace PV {

class LogLatWTAFunction : public LayerFunction {
public:
	LogLatWTAFunction(const char * name);

    virtual pvdata_t evaluate(float time, HyPerLayer * l);

}; // end class LogWTAFunction

}  // end namespace PV

#endif /* LOGLATWTAFUNCTION_HPP_ */
