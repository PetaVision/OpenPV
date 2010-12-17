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
	relaxation = parent->parameters()->value(name, "relaxation", 1.0);
	return EXIT_SUCCESS;
}  // end of GenerativeLayer::initialize()

int GenerativeLayer::updateV() {
   pvdata_t * V = getV();
   pvdata_t ** phi = getCLayer()->phi;
   pvdata_t * phiExc = phi[PHI_EXC];
   pvdata_t * phiInh = phi[PHI_INH];
   for( int k=0; k<getNumNeurons(); k++ ) {
      V[k] += relaxation*(phiExc[k] - phiInh[k] - sparsitytermderivative(V[k]));
      // Thresholding would go here
   }
   return EXIT_SUCCESS;
}  // end of GenerativeLayer::updateV()


}  // end of namespace PV block

