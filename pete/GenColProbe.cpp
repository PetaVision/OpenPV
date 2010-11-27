/*
 * GenColProbe.cpp
 *
 *  Created on: Nov 26, 2010
 *      Author: pschultz
 */

#include "GenColProbe.hpp"

namespace PV {

GenColProbe::GenColProbe() : ColProbe() {
    numTerms = 0;
    terms = NULL;
    layers = NULL;
}  // end GenColProbe::GenColProbe()

GenColProbe::GenColProbe(const char * filename) : ColProbe(filename) {
    numTerms = 0;
    terms = NULL;
}  // end GenColProbe::GenColProbe(const char *)

GenColProbe::~GenColProbe() {
    if( numTerms ) free(terms);
}  // end GenColProbe::~GenColProbe()

int GenColProbe::addTerm(LayerFunctionProbe * p, HyPerLayer * l) {
    LayerFunctionProbe ** newterms = (LayerFunctionProbe **) malloc( (numTerms+1)*sizeof(LayerFunctionProbe *) );
    if( !newterms ) return EXIT_FAILURE;
    HyPerLayer ** newlayers = (HyPerLayer **) malloc( (numTerms+1)*sizeof(HyPerLayer *) );
    if( !newlayers ) {
    	free(newterms);
        return EXIT_FAILURE;
    }
    for( int n=0; n<numTerms; n++) {
        newterms[n] = terms[n];
        newlayers[n] = layers[n];
    }
    newterms[numTerms] = p;
    newlayers[numTerms] = l;
    free(terms);
    free(layers);
    terms = newterms;
    layers = newlayers;
    numTerms++;
    return EXIT_SUCCESS;
}  // end GenColProbe::addTerm(LayerFunctionProbe *, HyPerLayer *)

pvdata_t GenColProbe::evaluate(float time) {
    pvdata_t sum = 0;
    for( int n=0; n<numTerms; n++) {
        sum += terms[n]->getFunction()->evaluate(time, layers[n]);
    }
    return sum;
}  // end GenColProbe::evaluate(float)

int GenColProbe::outputState(float time, HyPerCol * hc) {
    printf("time = %f, %s = %f\n", time, hc->getName(), evaluate(time));
    return EXIT_SUCCESS;
}  // end GenColProbe::outputState(float)

}  // end namespace PV
