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

GenerativeLayer::GenerativeLayer(const char * name, HyPerCol * hc) : GV1(name, hc) {
	initialize(TypeV1Simple);
}  // end of GenerativeLayer::GenerativeLayer(const char *, HyperCol *)

GenerativeLayer::GenerativeLayer(const char * name, HyPerCol * hc, PVLayerType type) : GV1(name, hc, type){
    initialize(type);
}  // end of GenerativeLayer::GenerativeLayer(const char *, HyperCol *, PVLayerType *)

int GenerativeLayer::initialize(PVLayerType type) {
	relaxation = parent->parameters()->value(name, "relaxation", 1.0);
	return EXIT_SUCCESS;
}  // end of GenerativeLayer::initialize(PVLayerType)

int GenerativeLayer::updateState(float time, float dt) {
	   pv_debug_info("[%d]: GenerativeLayer::updateState:", clayer->columnId);

	   // Update V using dV/dt = relaxationrate*(clayer->phi[0]-clayer->phi[1])
	   // clayer->activity->data to new V
	   // zero out clayer->phi[0] and clayer->phi[1]

	   const int nx = clayer->loc.nx;
	   const int ny = clayer->loc.ny;
	   const int nf = clayer->numFeatures;
	   const int marginWidth = clayer->loc.nPad;

	   pvdata_t * V = getV();
	   pvdata_t * phiExc   = clayer->phi[PHI_EXC];
	   pvdata_t * phiInh   = clayer->phi[PHI_INH];
	   pvdata_t * activity = clayer->activity->data;

	   // make sure activity in border is zero
	   //
	   for (int k = 0; k < getNumExtended(); k++) {
	      activity[k] = 0.0;
	   }

	   for (int k = 0; k < getNumNeurons(); k++) {
	      int kex = kIndexExtended(k, nx, ny, nf, marginWidth);
	      V[k] += relaxation*(phiExc[k] - phiInh[k] - sparsitytermderivative(V[k])); // essentially only change from V1
	      activity[kex] = V[k];

	      // reset accumulation buffers
	      phiExc[k] = 0.0;
	      phiInh[k] = 0.0;
	   }

	   return EXIT_SUCCESS;
}  // end of GenerativeLayer::updateState(float, float)

}  // end of namespace PV block

