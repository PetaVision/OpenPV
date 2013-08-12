/*
 * MatchingPursuitProbe.cpp
 *
 *  Created on: Aug 12, 2013
 *      Author: pschultz
 */

#include "MatchingPursuitProbe.hpp"

namespace PV {

MatchingPursuitProbe::MatchingPursuitProbe(const char * name, HyPerCol * hc) {
   initMatchingPursuitProbe_base();
   initMatchingPursuitProbe(name, hc);
}

MatchingPursuitProbe::MatchingPursuitProbe() {
}

int MatchingPursuitProbe::initMatchingPursuitProbe_base() {
   return PV_SUCCESS;
}

int MatchingPursuitProbe::initMatchingPursuitProbe(const char * name, HyPerCol * hc) {
   if (hc==NULL) {
      fprintf(stderr, "MatchingPursuitProbe error: HyPerCol argument cannot be null.\n");
   }
   if (name==NULL) {
      if (hc->columnId()==0) {
         fprintf(stderr, "MatchingPursuitProbe error: argument \"name\" cannot be null.\n");
      }
      MPI_Barrier(hc->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   const char * filename = hc->parameters()->stringValue(name, "probeOutputFile");
   const char * target_layer_name = hc->parameters()->stringValue(name, "targetLayer");
   if (target_layer_name==NULL) {
      if (hc->columnId()==0) {
         fprintf(stderr, "MatchingPursuitProbe \"%s\" error: targetLayer parameter must be set.\n", name);
      }
      MPI_Barrier(hc->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   HyPerLayer * target_layer = hc->getLayerFromName(target_layer_name);
   if (target_layer==NULL) {
      if (hc->columnId()==0) {
         fprintf(stderr, "MatchingPursuitProbe \"%s\" error: targetLayer \"%s\" is not a valid HyPerLayer in the column.\n", name, target_layer_name);
      }
      MPI_Barrier(hc->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   initLayerProbe(filename, target_layer);
   return PV_SUCCESS;
}

int MatchingPursuitProbe::outputState(double timed) {
   const PVLayerLoc * loc = getTargetLayer()->getLayerLoc();
   if (timed>0.0) {
      for (int k=0; k<getTargetLayer()->getNumNeurons(); k++) {
         int kGlobal = globalIndexFromLocal(k, *loc);
         pvdata_t correctValue = (double) kGlobal + nearbyint(timed) < 254.5 ? 0.0f : (pvdata_t) kGlobal/255.0f;
         int kExtended = kIndexExtended(k, loc->nx, loc->ny, loc->nf, loc->nb);
         assert(fabs(getTargetLayer()->getLayerData()[kExtended]-correctValue)<1e-7);
      }
   }
   return PV_SUCCESS;
}

MatchingPursuitProbe::~MatchingPursuitProbe() {
}

} /* namespace PV */
