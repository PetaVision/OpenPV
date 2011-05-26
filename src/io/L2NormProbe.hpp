/*
 * L2NormProbe.hpp
 *
 *  Created on: Nov 19, 2010
 *      Author: pschultz
 */

#ifndef L2NORMPROBE_HPP_
#define L2NORMPROBE_HPP_

#include "LayerFunctionProbe.hpp"
#include "L2NormFunction.hpp"

namespace PV {
class L2NormProbe : public LayerFunctionProbe {
public:
	L2NormProbe(const char * msg);
	L2NormProbe(const char * filename, HyPerCol * hc, const char * msg);
	~L2NormProbe();
    int outputState(float time, HyPerLayer * l);
    pvdata_t evaluate();

}; // end class L2NormProbe

}  // end namespace PV

#endif /* L2NORMPROBE_HPP_ */
