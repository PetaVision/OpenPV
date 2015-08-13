/*
 * GenColProbe.cpp
 *
 *  Created on: Nov 26, 2010
 *      Author: pschultz
 */

#include "GenColProbe.hpp"
#include "ConnFunctionProbe.hpp"
#include "LayerFunctionProbe.hpp"

namespace PV {

GenColProbe::GenColProbe() : ColProbe() { // Default constructor to be called by derived classes.
   // They should call GenColProbe::initialize from their own initialization routine
   // instead of calling a non-default constructor.
   initialize_base();
}  // end GenColProbe::GenColProbe(const char *)

GenColProbe::GenColProbe(const char * probename, HyPerCol * hc) : ColProbe() {
   initialize_base();
   initializeGenColProbe(probename, hc);
}  // end GenColProbe::GenColProbe(const char *, const char *, HyPerCol *)

GenColProbe::~GenColProbe() {
   if( numLayerTerms ) {
      free(layerTerms);
   }
}  // end GenColProbe::~GenColProbe()

int GenColProbe::initialize_base() {
   numLayerTerms = 0;
   layerTerms = NULL;
   numConnTerms = 0;
   connTerms = NULL;

   return PV_SUCCESS;
}

int GenColProbe::initializeGenColProbe(const char * probename, HyPerCol * hc) {
   return ColProbe::initialize(probename, hc);
}

int GenColProbe::addLayerTerm(LayerFunctionProbe * p, HyPerLayer * l, pvdata_t coeff) {
   gencolprobelayerterm * newtheterms = (gencolprobelayerterm *) malloc( (numLayerTerms+1)*sizeof(gencolprobelayerterm) );
   if( !newtheterms ) return PV_FAILURE;
   for( int n=0; n<numLayerTerms; n++) {
      newtheterms[n] = layerTerms[n];
   }
   newtheterms[numLayerTerms].function = p;
   newtheterms[numLayerTerms].layer = l;
   newtheterms[numLayerTerms].coeff = coeff;
   free(layerTerms);
   layerTerms = newtheterms;
   numLayerTerms++;
   return PV_SUCCESS;
}  // end GenColProbe::addTerm(LayerFunctionProbe *, HyPerLayer *)

int GenColProbe::addConnTerm(ConnFunctionProbe * p, BaseConnection * c, pvdata_t coeff) {
   gencolprobeconnterm * newtheterms = (gencolprobeconnterm *) malloc( (numConnTerms+1)*sizeof(gencolprobeconnterm) );
   if( !newtheterms ) return PV_FAILURE;
   for( int n=0; n<numConnTerms; n++ ) {
      newtheterms[n] = connTerms[n];
   }
   newtheterms[numConnTerms].function = p;
   newtheterms[numConnTerms].conn = c;
   newtheterms[numConnTerms].coeff = coeff;
   free(connTerms);
   connTerms = newtheterms;
   numConnTerms++;
   return PV_SUCCESS;
}

pvdata_t GenColProbe::evaluate(double timef, int batchIdx) {
   pvdata_t sum = 0;
   for( int n=0; n<numLayerTerms; n++) {
      gencolprobelayerterm thisterm = layerTerms[n];
      sum += thisterm.coeff*( (thisterm.function)->getFunction()->evaluate(timef, thisterm.layer, batchIdx) );
   }
   for( int n=0; n<numConnTerms; n++) {
      gencolprobeconnterm thisterm = connTerms[n];
      sum += thisterm.coeff*( (thisterm.function)->evaluate(timef) );
   }
   return sum;
}  // end GenColProbe::evaluate(float)

int GenColProbe::outputState(double time, HyPerCol * hc) {
   for(int b = 0; b < hc->getNBatch(); b++){
      pvdata_t colprobeval = evaluate(time, b);
#ifdef PV_USE_MPI
      if( hc->icCommunicator()->commRank() != 0 ) return PV_SUCCESS;
#endif // PV_USE_MPI
      fprintf(stream->fp, "time = %f, b = %d, %s = %f\n", time, b, hc->getName(), colprobeval);
   }
   fflush(stream->fp);
   return PV_SUCCESS;
}  // end GenColProbe::outputState(float, HyPerCol *)

}  // end namespace PV
