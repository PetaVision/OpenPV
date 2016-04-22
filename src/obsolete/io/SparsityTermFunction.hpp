/*
 * SparsityTermFunction.hpp
 *
 *  Created on: Nov 26, 2010
 *      Author: pschultz
 */

#ifndef SPARSITYTERMFUNCTION_HPP_
#define SPARSITYTERMFUNCTION_HPP_

#include "LayerFunction.hpp"

namespace PV {

class SparsityTermFunction : public LayerFunction {
public:
	SparsityTermFunction(const char * name);

    virtual pvdata_t evaluate(float time, HyPerLayer * l);

}; // end class SparsityTermFunction

}  // end namespace PV

#endif /* SPARSITYTERMFUNCTION_HPP_ */
