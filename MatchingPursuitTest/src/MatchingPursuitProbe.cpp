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
#ifdef PV_USE_MPI
      MPI_Barrier(hc->icCommunicator()->communicator());
#endif
      exit(EXIT_FAILURE);
   }
   initLayerProbe(name, hc);
   return PV_SUCCESS;
}

int MatchingPursuitProbe::outputState(double timed) {
   int status = PV_SUCCESS;
   const PVLayerLoc * loc = getTargetLayer()->getLayerLoc();
   if (timed>0.0) {
      for (int k=0; k<getTargetLayer()->getNumNeurons(); k++) {
         int kGlobal = globalIndexFromLocal(k, *loc);
         pvdata_t correctValue = nearbyint((double)kGlobal + timed)==256.0 ? (pvdata_t) kGlobal/255.0f : 0.0f;
         int kExtended = kIndexExtended(k, loc->nx, loc->ny, loc->nf, loc->nb);
         pvdata_t observed = getTargetLayer()->getLayerData()[kExtended];
         pvdata_t relerr = fabs(observed-correctValue)/correctValue;
         if (relerr>1e-7) {
            fprintf(stderr, "Time %f: Neuron %d (global index) has relative error %f (%f versus correct %f)\n", timed, kGlobal, relerr, observed, correctValue);
            status = PV_FAILURE;
         }
      }
   }
   assert(status==PV_SUCCESS);
   return status;
}

MatchingPursuitProbe::~MatchingPursuitProbe() {
}

} /* namespace PV */
