/*
 * GenColProbe.cpp
 *
 *  Created on: Nov 26, 2010
 *      Author: pschultz
 */

#include "GenColProbe.hpp"

namespace PV {

GenColProbe::GenColProbe(const char * probename) : ColProbe() {
   initialize_base();
}  // end GenColProbe::GenColProbe(const char *)

GenColProbe::GenColProbe(const char * probename, const char * filename, HyPerCol * hc) : ColProbe() {
   initialize_base();
   initializeGenColProbe(probename, filename, hc);
}  // end GenColProbe::GenColProbe(const char *, const char *, HyPerCol *)

GenColProbe::~GenColProbe() {
   if( numLayerTerms ) {
      free(layerTerms);
   }
}  // end GenColProbe::~GenColProbe()

int GenColProbe::initialize_base() {
   numLayerTerms = 0;
   layerTerms = NULL;

   return PV_SUCCESS;
}

int GenColProbe::initializeGenColProbe(const char * probename, const char * filename, HyPerCol * hc) {
   return ColProbe::initialize(probename, filename, hc);
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

int GenColProbe::addConnTerm(ConnFunctionProbe * p, HyPerConn * c, pvdata_t coeff) {
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

pvdata_t GenColProbe::evaluate(float timef) {
   pvdata_t sum = 0;
   for( int n=0; n<numLayerTerms; n++) {
      gencolprobelayerterm thisterm = layerTerms[n];
      sum += thisterm.coeff*( (thisterm.function)->getFunction()->evaluate(timef, thisterm.layer) );
   }
   for( int n=0; n<numConnTerms; n++) {
      gencolprobeconnterm thisterm = connTerms[n];
      sum += thisterm.coeff*( (thisterm.function)->evaluate(timef) );
   }
   return sum;
}  // end GenColProbe::evaluate(float)

int GenColProbe::outputState(float time, HyPerCol * hc) {
   pvdata_t colprobeval = evaluate(time);
#ifdef PV_USE_MPI
   if( hc->icCommunicator()->commRank() != 0 ) return PV_SUCCESS;
#endif // PV_USE_MPI
   fprintf(fp, "time = %f, %s = %f\n", time, hc->getName(), colprobeval);
   return PV_SUCCESS;
}  // end GenColProbe::outputState(float)

int GenColProbe::writeState(float time, HyPerCol * hc, pvdata_t value) {
#ifdef PV_USE_MPI
   // In MPI mode, this function should only be called by the root processor.
   assert(hc->icCommunicator()->commRank() == 0);
#endif // PV_USE_MPI
   int printstatus = fprintf(fp, "time = %f, %s = %f\n", time, hc->getName(), evaluate(time));
   return printstatus > 0 ? PV_SUCCESS : PV_FAILURE;
}

}  // end namespace PV
