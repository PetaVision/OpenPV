/*
 * L2NormFunction.hpp
 *
 *  Created on: Nov 26, 2010
 *      Author: pschultz
 */

#ifndef L2NORMFUNCTION_HPP_
#define L2NORMFUNCTION_HPP_

#include "LayerFunction.hpp"

namespace PV {

class L2NormFunction : public LayerFunction {
public:
    L2NormFunction(const char * name);

    virtual pvdata_t evaluateLocal(float time, HyPerLayer * l, int batchIdx);

}; // end class L2NormFunction

}  // end namespace PV

#endif /* L2NORMFUNCTION_HPP_ */
