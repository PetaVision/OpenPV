/*
 * GenerativeProbe.hpp
 *
 *  Created on: Nov 26, 2010
 *      Author: pschultz
 */

#ifndef GENCOLPROBE_HPP_
#define GENCOLPROBE_HPP_

#define DEFAULT_GENCOLPROBE_COEFFICIENT ((pvdata_t) 1)

#include "ColProbe.hpp"
#include "../columns/HyPerCol.hpp"
#include "../layers/HyPerLayer.hpp"
#include "LayerFunctionProbe.hpp"

namespace PV {

typedef struct gencolprobeterm_ {
    LayerFunctionProbe * function;
    HyPerLayer * layer;
    pvdata_t coeff;
} gencolprobeterm;

class GenColProbe : public ColProbe {
public:
	GenColProbe(const char * name);
	GenColProbe(const char * probename, const char * filename, HyPerCol * hc);
	~GenColProbe();
	int initialize_base();

	int addTerm(LayerFunctionProbe * p, HyPerLayer * l);
	int addTerm(LayerFunctionProbe * p, HyPerLayer * l, pvdata_t coeff);
	virtual pvdata_t evaluate(float time);
	virtual int outputState(float time, HyPerCol * hc);
	virtual int writeState(float time, HyPerCol * hc, pvdata_t value);

protected:
	int numTerms;
	gencolprobeterm * terms;

}; // end class GenColProbe

}  // end namespace PV

#endif /* GENCOLPROBE_HPP_ */
