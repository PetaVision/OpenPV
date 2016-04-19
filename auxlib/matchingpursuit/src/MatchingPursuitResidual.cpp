/*
 * MatchingPursuitResidual.cpp
 *
 *  Created on: Aug 13, 2013
 *      Author: pschultz
 */

#include "MatchingPursuitResidual.hpp"

namespace PVMatchingPursuit {

MatchingPursuitResidual::MatchingPursuitResidual(const char * name, PV::HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

MatchingPursuitResidual::MatchingPursuitResidual() {
   initialize_base();
}

int MatchingPursuitResidual::initialize_base() {
   inputInV = false;
   return PV_SUCCESS;
}

int MatchingPursuitResidual::initialize(const char * name, PV::HyPerCol * hc) {
   int status = PV::ANNLayer::initialize(name, hc);
   return status;
}

bool MatchingPursuitResidual::needUpdate(double time, double dt) {
   // TODO: account for delays, phases and triggerOffset in determining time to trigger
   if (triggerLayer && triggerLayer->getLastUpdateTime() > this->getLastUpdateTime()) {
      inputInV = false;
   }
   return true;
}

int MatchingPursuitResidual::updateState(double timed, double dt) {
   pvdata_t * V = getV();
   if (inputInV) {
      for (int k=0; k<getNumNeuronsAllBatches(); k++) {
         V[k] -= GSyn[1][k];
      }
   }
   else {
      for (int k=0; k<getNumNeuronsAllBatches(); k++) {
         V[k] = GSyn[0][k];
      }
      inputInV = true;
   }
   PVLayerLoc const * loc = getLayerLoc();
   setActivity_HyPerLayer(loc->nbatch, getNumNeurons(), getActivity(), V, loc->nx, loc->ny, loc->nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
   return PV_SUCCESS;
}

MatchingPursuitResidual::~MatchingPursuitResidual() {
}

PV::BaseObject * createMatchingPursuitResidual(char const * name, PV::HyPerCol * hc) {
   return hc ? new MatchingPursuitResidual(name, hc) : NULL;
}

} /* namespace PVMatchingPursuit */
