/*
 * LogLatWTAGenLayer.cpp
 *
 *  Created on: Apr 20, 2011
 *      Author: peteschultz
 */

#include "LogLatWTAGenLayer.hpp"

namespace PV {

LogLatWTAGenLayer::LogLatWTAGenLayer() {
   initialize_base();
}

LogLatWTAGenLayer::LogLatWTAGenLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

LogLatWTAGenLayer::~LogLatWTAGenLayer() {}

int LogLatWTAGenLayer::initialize_base() {
   return PV_SUCCESS;
}

int LogLatWTAGenLayer::updateState(double timef, double dt) {
   int status;
   status = doUpdateState(timef, dt, getLayerLoc(), getCLayer()->activity->data, getV(), getNumChannels(), GSyn[0], sparsitytermderivative, dV, AMax, AMin, VThresh, VWidth, AShift, relaxation, auxChannelCoeff, sparsityTermCoeff, persistence, activityThreshold);
   return status;
}

int LogLatWTAGenLayer::doUpdateState(double timef, double dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int num_channels, pvdata_t * gSynHead, pvdata_t * sparsitytermderivative, pvdata_t * dAold, pvdata_t AMax, pvdata_t AMin, pvdata_t VThresh, pvdata_t AShift, pvdata_t VWidth, pvdata_t relaxation, pvdata_t auxChannelCoeff, pvdata_t sparsityTermCoeff, pvdata_t persistence, pvdata_t activity_threshold) {
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   pvdata_t relax_remaining = relaxation;
   pvdata_t trunc_rel = relaxation;
   while(relax_remaining > 0 && trunc_rel > relaxation*1e-6) {
      updateSparsityTermDeriv_LogLatWTAGenLayer(num_neurons, getLayerLoc()->nf, V, sparsitytermderivative);
      update_dV_GenerativeLayer(num_neurons, V, gSynHead, sparsitytermderivative, dAold, AMax, AMin, VThresh, relaxation, auxChannelCoeff, sparsityTermCoeff, persistence);
      trunc_rel = reduce_relaxation(num_neurons, V, dV, relax_remaining);
      updateV_GenerativeLayer(num_neurons, V, dV, A, AMax, AMin, VThresh, AShift, VWidth, trunc_rel, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
      relax_remaining -=trunc_rel;
   }
   setActivity_GenerativeLayer(num_neurons, A, V, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up, activity_threshold); // setActivity();
   resetGSynBuffers_HyPerLayer(num_neurons, getNumChannels(), gSynHead); // resetGSynBuffers();
   return PV_SUCCESS;
}

}  // end of namespace PV block
