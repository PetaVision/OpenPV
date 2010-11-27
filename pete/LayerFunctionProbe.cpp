/*
 * LayerFunctionProbe.cpp
 *
 *  Created on: Nov 26, 2010
 *      Author: pschultz
 */

#include "LayerFunctionProbe.hpp"

namespace PV {

LayerFunctionProbe::LayerFunctionProbe(const char * msg) :
    StatsProbe(BufV, msg) {
    function = NULL;
}  // end LayerFunctionProbe::LayerFunctionProbe(const char *)

LayerFunctionProbe::LayerFunctionProbe(const char * filename, const char * msg) :
    StatsProbe(filename, BufV, msg) {
    function = NULL;
}  // end LayerFunctionProbe::LayerFunctionProbe(const char *, const char *)

LayerFunctionProbe::LayerFunctionProbe(const char * msg, LayerFunction * F) :
    StatsProbe(BufV, msg) {
    function = F;
}  // end LayerFunctionProbe::LayerFunctionProbe(const char *, LayerFunction *)

LayerFunctionProbe::LayerFunctionProbe(const char * filename, const char * msg, LayerFunction * F) :
    StatsProbe(filename, BufV, msg) {
    function = F;
}  // end LayerFunctionProbe::LayerFunctionProbe(const char *, const char *, LayerFunction *)

int LayerFunctionProbe::outputState(float time, HyPerLayer * l) {
    if( function ) {
        pvdata_t val = function->evaluate(time, l);
        fprintf(fp, "%st = %6.3f numNeurons = %8d Value            = %f\n", msg, time, l->getNumNeurons(), val);
        fflush(fp);
        return EXIT_SUCCESS;
    }
    else {
    	fprintf(stderr, "LayerFunctionProbe %lu for layer %s: function has not been set\n", this, l->getName());
        return EXIT_FAILURE;
    }
}  // end LayerFunctionProbe::outputState(float, HyPerLayer *)

}  // end namespace PV
