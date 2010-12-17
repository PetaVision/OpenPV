/*
 * GenerativeProbe.hpp
 *
 *  Created on: Nov 26, 2010
 *      Author: pschultz
 */

#ifndef GENCOLPROBE_HPP_
#define GENCOLPROBE_HPP_

#define DEFAULT_GENCOLPROBE_COEFFICIENT ((pvdata_t) 1.0)

#include "../PetaVision/src/io/ColProbe.hpp"
#include "../PetaVision/src/columns/HyPerCol.hpp"
#include "../PetaVision/src/layers/HyPerLayer.hpp"
#include "LayerFunctionProbe.hpp"

typedef struct gencolprobeterm_ {
    void * function; // LayerFunctionProbe * function;
    void * layer; // HyPerLayer * layer;
    pvdata_t coeff;
} gencolprobeterm;

namespace PV {
class GenColProbe : public ColProbe {
public:
	GenColProbe();
	GenColProbe(const char * filename);
	~GenColProbe();
	int initialize_base();

	int addTerm(LayerFunctionProbe * p, HyPerLayer * l);
	int addTerm(LayerFunctionProbe * p, HyPerLayer * l, pvdata_t coeff);
	virtual pvdata_t evaluate(float time);
	virtual int outputState(float time, HyPerCol * hc);

protected:
	int numTerms;
	gencolprobeterm * terms;


}; // end class GenColProbe

}  // end namespace PV

#endif /* GENCOLPROBE_HPP_ */
