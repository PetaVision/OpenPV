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
   LayerProbe::initialize(name, hc);
   return PV_SUCCESS;
}

int MatchingPursuitProbe::initNumValues() {
   return setNumValues(1); /* MatchingPursuitProbe has not been generalized for batches. */
}

int MatchingPursuitProbe::calcValues(double timevalue) {
   int status = PV_SUCCESS;
   pvdata_t maxrelerr = (pvdata_t) 0;
   if (timevalue>0.0) {
      const PVLayerLoc * loc = getTargetLayer()->getLayerLoc();
      for (int k=0; k<getTargetLayer()->getNumNeurons(); k++) {
         int kGlobal = globalIndexFromLocal(k, *loc);
         pvdata_t correctValue = nearbyint((double)kGlobal + timevalue)==256.0 ? (pvdata_t) kGlobal/255.0f : 0.0f;
         int kExtended = kIndexExtended(k, loc->nx, loc->ny, loc->nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
         pvdata_t observed = getTargetLayer()->getLayerData()[kExtended];
         pvdata_t relerr = fabs(observed-correctValue)/correctValue;
         if (maxrelerr<relerr) { maxrelerr = relerr; }
         if (relerr>1e-7) {
            fprintf(stderr, "Time %f: Neuron %d (global index) has relative error %f (%f versus correct %f)\n", timevalue, kGlobal, relerr, observed, correctValue);
            status = PV_FAILURE;
         }
      }
   }
   getValuesBuffer()[0] = maxrelerr;
   return status;
}

int MatchingPursuitProbe::outputState(double timed) {
   int status = calcValues(timed);
   assert(status==PV_SUCCESS);
   return status;
}

MatchingPursuitProbe::~MatchingPursuitProbe() {
}

PV::BaseObject * createMatchingPursuitProbe(char const * name, PV::HyPerCol * hc) {
   return hc ? new MatchingPursuitProbe(name, hc) : NULL;
}

} /* namespace PV */
