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

namespace PV {

class ConnFunctionProbe;
class LayerFunctionProbe;

typedef struct gencolprobelayerterm_ {
   LayerFunctionProbe * function;
   HyPerLayer * layer;
   pvdata_t coeff;
} gencolprobelayerterm;

typedef struct gencolprobconnterm_ {
   ConnFunctionProbe * function;
   BaseConnection * conn;
   pvdata_t coeff;
} gencolprobeconnterm;

class GenColProbe : public ColProbe {
public:
   GenColProbe(const char * probename, HyPerCol * hc);
   ~GenColProbe();

   int addConnTerm(ConnFunctionProbe * p, BaseConnection * c, pvdata_t coeff);
   int addLayerTerm(LayerFunctionProbe * p, HyPerLayer * l, pvdata_t coeff);
   virtual int outputState(double time, HyPerCol * hc);

protected:
   GenColProbe();
   int initializeGenColProbe(const char * probename, HyPerCol * hc);
   virtual pvdata_t evaluate(double timef, int batchIdx);

   int numLayerTerms;
   gencolprobelayerterm * layerTerms;

   int numConnTerms;
   gencolprobeconnterm * connTerms;

private:
   int initialize_base();

}; // end class GenColProbe

}  // end namespace PV

#endif /* GENCOLPROBE_HPP_ */
