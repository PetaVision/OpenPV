/*
 * SparsityTermProbe.hpp
 *
 *  Created on: Nov 18, 2010
 *      Author: pschultz
 */

#ifndef SPARSITYTERMPROBE_HPP_
#define SPARSITYTERMPROBE_HPP_

#include "LayerFunctionProbe.hpp"
#include "SparsityTermFunction.hpp"

namespace PV {

class SparsityTermProbe : public LayerFunctionProbe {
public:
	SparsityTermProbe(const char * msg);
	SparsityTermProbe(const char * filename, HyPerCol * hc, const char * msg);
	~SparsityTermProbe();
    virtual int outputState(float time, HyPerLayer * l);
};

}  // end namespace PV

#endif /* SPARSITYTERMPROBE_HPP_ */
