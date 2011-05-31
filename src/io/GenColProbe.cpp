/*
 * GenColProbe.cpp
 *
 *  Created on: Nov 26, 2010
 *      Author: pschultz
 */

#include "GenColProbe.hpp"

namespace PV {

GenColProbe::GenColProbe(const char * probename) : ColProbe(probename) {
    initialize_base();
}  // end GenColProbe::GenColProbe()

GenColProbe::GenColProbe(const char * probename, const char * filename, HyPerCol * hc) : ColProbe(probename, filename, hc) {
    initialize_base();
}  // end GenColProbe::GenColProbe(const char *)

int GenColProbe::initialize_base() {
    numTerms = 0;
    terms = NULL;

    return EXIT_SUCCESS;
}

GenColProbe::~GenColProbe() {
    if( numTerms ) {
        free(terms);
    }
}  // end GenColProbe::~GenColProbe()

int GenColProbe::addTerm(LayerFunctionProbe * p, HyPerLayer * l) {
    return addTerm(p, l, DEFAULT_GENCOLPROBE_COEFFICIENT);
}

int GenColProbe::addTerm(LayerFunctionProbe * p, HyPerLayer * l, pvdata_t coeff) {
    gencolprobeterm * newtheterms = (gencolprobeterm *) malloc( (numTerms+1)*sizeof(gencolprobeterm) );
    if( !newtheterms ) return EXIT_FAILURE;
    for( int n=0; n<numTerms; n++) {
        newtheterms[n] = terms[n];
    }
    newtheterms[numTerms].function = p;
    newtheterms[numTerms].layer = l;
    newtheterms[numTerms].coeff = coeff;
    free(terms);
    terms = newtheterms;
    numTerms++;
    return EXIT_SUCCESS;
}  // end GenColProbe::addTerm(LayerFunctionProbe *, HyPerLayer *)

pvdata_t GenColProbe::evaluate(float time) {
    pvdata_t sum = 0;
    for( int n=0; n<numTerms; n++) {
    	gencolprobeterm thisterm = terms[n];
        sum += thisterm.coeff*( (thisterm.function)->getFunction()->evaluate(time, thisterm.layer) );
    }
    return sum;
}  // end GenColProbe::evaluate(float)

int GenColProbe::outputState(float time, HyPerCol * hc) {
    fprintf(fp, "time = %f, %s = %f\n", time, hc->getName(), evaluate(time));
    return EXIT_SUCCESS;
}  // end GenColProbe::outputState(float)

}  // end namespace PV
