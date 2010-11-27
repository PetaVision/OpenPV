/*
 * GenerativeProbe.hpp
 *
 *  Created on: Nov 26, 2010
 *      Author: pschultz
 */

#ifndef GENCOLPROBE_HPP_
#define GENCOLPROBE_HPP_

#include "../PetaVision/src/io/ColProbe.hpp"
#include "../PetaVision/src/columns/HyPerCol.hpp"
#include "../PetaVision/src/layers/HyPerLayer.hpp"
#include "LayerFunctionProbe.hpp"

namespace PV {
class GenColProbe : public ColProbe {
public:
	GenColProbe();
	GenColProbe(const char * filename);
	~GenColProbe();

	int addTerm(LayerFunctionProbe * p, HyPerLayer * l);
	virtual pvdata_t evaluate(float time);
	virtual int outputState(float time, HyPerCol * hc);

protected:
	int numTerms;
	LayerFunctionProbe ** terms;
	HyPerLayer ** layers;


}; // end class GenColProbe

}  // end namespace PV

#endif /* GENCOLPROBE_HPP_ */
