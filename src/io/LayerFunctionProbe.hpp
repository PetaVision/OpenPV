/*
 * LayerFunctionProbe.hpp
 *
 *  Created on: Nov 26, 2010
 *      Author: pschultz
 */

#ifndef LAYERFUNCTIONPROBE_HPP_
#define LAYERFUNCTIONPROBE_HPP_

#include "StatsProbe.hpp"
#include "../layers/HyPerLayer.hpp"
#include "LayerFunction.hpp"

namespace PV {

class LayerFunctionProbe : public StatsProbe {
public:
	LayerFunctionProbe(const char * msg);
	LayerFunctionProbe(const char * filename, HyPerCol * hc, const char * msg);
	LayerFunctionProbe(const char * msg, LayerFunction * F);
	LayerFunctionProbe(const char * filename, HyPerCol * hc, const char * msg, LayerFunction * F);

	LayerFunction * getFunction() {return function;}
	void setFunction(LayerFunction * f) {function = f;}
	virtual int outputState(float time, HyPerLayer * l);

protected:
	LayerFunction * function;
};

}  // end namespace PV

#endif /* LAYERFUNCTIONPROBE_HPP_ */
