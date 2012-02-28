/*
 * ReciprocalEnergyProbe.cpp
 *
 *  Created on: Feb 17, 2012
 *      Author: pschultz
 */

#include "ReciprocalEnergyProbe.hpp"

namespace PV {

ReciprocalEnergyProbe::ReciprocalEnergyProbe() {
   initialize_base();
}

ReciprocalEnergyProbe::ReciprocalEnergyProbe(const char * probename, const char * filename, HyPerCol * hc) {
   initialize_base();
   initialize(name, filename, hc);
}

int ReciprocalEnergyProbe::initialize_base() {
   return PV_SUCCESS;
}

int ReciprocalEnergyProbe::initialize(const char * probename, const char * filename, HyPerCol * hc) {
   int status = BaseConnectionProbe::initialize(probename, filename, hc);
   return status;
}

int ReciprocalEnergyProbe::outputState(float timef, HyPerConn * c) {
   int status = PV_SUCCESS;
   ReciprocalConn * rc = dynamic_cast<ReciprocalConn *>(c);
   if( rc == NULL ) {
      fprintf(stderr, "ReciprocalEnergyProbe \"%s\": connection \"%s\" is not a ReciprocalConn.\n", name, c->getName());
      status = PV_FAILURE;
      abort();
   }
   if( status == PV_SUCCESS ) {
      double energy = 0.0f;
      for( int arbor=0; arbor<rc->numberOfAxonalArborLists(); arbor++) {
         for( int k=0; k<rc->numDataPatches(); k++) {
            PVPatch * p = rc->getWeights(k, arbor); // getKernelPatch(arbor, k);
            // const pvdata_t * wdata = p->data;
            pvdata_t * wdata = rc->get_wDataStart(arbor) + k*rc->xPatchSize()*rc->yPatchSize()*rc->fPatchSize() + p->offset;
            short int nx = p->nx;
            short int ny = p->ny;
            for( int n=0; n<nx*ny*rc->fPatchSize(); n++ ) {
               int f = featureIndex(n,nx,ny,rc->fPatchSize());
               ReciprocalConn * reciprocalWgts = rc->getReciprocalWgts();
               PVPatch * recipPatch = reciprocalWgts->getWeights(f,arbor);
               pvdata_t * recipDataStart = reciprocalWgts->get_wDataStart(arbor);
               const pvdata_t * recipwdata = recipDataStart+f*reciprocalWgts->xPatchSize()*reciprocalWgts->yPatchSize()*reciprocalWgts->fPatchSize()+recipPatch->offset;
               // const pvdata_t * recipwdata = rc->getReciprocalWgts()->getKernelPatch(arbor, f)->data;
               double wgtdiff = rc->getReciprocalFidelityCoeff()*(wdata[n] - ((double) rc->fPatchSize())/((double) rc->getReciprocalWgts()->fPatchSize())*recipwdata[k]);
               energy += wgtdiff*wgtdiff;
            }
         }
      }
      energy *= 0.5;
      fprintf(fp, "Time %f: Energy %f\n", timef, energy);
   }
   return status;
}

ReciprocalEnergyProbe::~ReciprocalEnergyProbe() {
}

} /* namespace PV */
