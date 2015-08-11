/*
 * L1NormFunction.hpp
 *
 *  Created on: Aug 11, 2015
 *      Author: peteschultz
 */

#ifndef L1NORMFUNCTION_HPP_
#define L1NORMFUNCTION_HPP_

#include <io/LayerFunction.hpp>

namespace PV {

class L1NormFunction : public LayerFunction {
public:
    L1NormFunction(const char * name);

    virtual pvdata_t evaluateLocal(float time, HyPerLayer * l, int batchIdx);

}; // end class L1NormFunction

}  // end namespace PV

#endif /* L1NORMFUNCTION_HPP_ */
