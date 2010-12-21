/*
 * GenerativeLayer.cpp
 *
 *  Created on: Oct 27, 2010
 *      Author: pschultz
 */

#include <assert.h>

#include "GenerativeConn.hpp"
#include "GenerativeLayer.hpp"

namespace PV {

GenerativeLayer::GenerativeLayer(const char * name, HyPerCol * hc) : V1(name, hc) {
	initialize();
}  // end of GenerativeLayer::GenerativeLayer(const char *, HyperCol *)

GenerativeLayer::GenerativeLayer(const char * name, HyPerCol * hc, PVLayerType type) : V1(name, hc, type){
    initialize();
}  // end of GenerativeLayer::GenerativeLayer(const char *, HyperCol *, PVLayerType *)

int GenerativeLayer::initialize() {
	PVParams * params = parent->parameters();
	relaxation = params->value(name, "relaxation", 1.0);
	activityThreshold = params->value(name, "activityThreshold", 0);
	return EXIT_SUCCESS;
}  // end of GenerativeLayer::initialize()

int GenerativeLayer::updateV() {
   pvdata_t * V = getV();
   pvdata_t ** phi = getCLayer()->phi;
   pvdata_t * phiExc = phi[PHI_EXC];
   pvdata_t * phiInh = phi[PHI_INH];
   for( int k=0; k<getNumNeurons(); k++ ) {
      V[k] += relaxation*(phiExc[k] - phiInh[k] - sparsitytermderivative(V[k]));
   }
   return EXIT_SUCCESS;
}  // end of GenerativeLayer::updateV()

int GenerativeLayer::setActivity() {
   const int nx = getLayerLoc()->nx;
   const int ny = getLayerLoc()->ny;
   const int nf = getCLayer()->numFeatures;
   const int marginWidth = getLayerLoc()->nPad;
   pvdata_t * activity = getCLayer()->activity->data;
   pvdata_t * V = getV();
   for( int k=0; k<getNumExtended(); k++ ) {
      activity[k] = 0;
   }
   for( int k=0; k<getNumNeurons(); k++ ) {
      int kex = kIndexExtended( k, nx, ny, nf, marginWidth );
      if( fabs(V[k]) > activityThreshold ) activity[kex] = V[k];
      // fabs(V[k]) > activityThreshold ? activity[kex] : 0;
   }
   return EXIT_SUCCESS;
}  // end of GenerativeLayer::setActivity()


}  // end of namespace PV block

