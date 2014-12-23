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

ReciprocalEnergyProbe::ReciprocalEnergyProbe(const char * probename, HyPerCol * hc) {
   initialize_base();
   int status = initialize(getName(), hc);
   assert(status==PV_SUCCESS);
}

int ReciprocalEnergyProbe::initialize_base() {
   return PV_SUCCESS;
}

int ReciprocalEnergyProbe::initialize(const char * probename, HyPerCol * hc) {
   int status = ConnFunctionProbe::initialize(probename, hc);
   return status;
}

int ReciprocalEnergyProbe::communicate() {
   int status = PV_SUCCESS;
   if(status==PV_SUCCESS) {
      targetRecipConn = dynamic_cast<ReciprocalConn *>(targetConn);
      if(targetRecipConn == NULL) {
         fprintf(stderr, "ReciprocalEnergyProbe \"%s\": connection \"%s\" is not a ReciprocalConn.\n", getName(), getTargetConn()->getName());
         status = PV_FAILURE;
      }
   }
   return status;
}

int ReciprocalEnergyProbe::allocateProbe() {
   return PV_SUCCESS;
}

double ReciprocalEnergyProbe::evaluate(double timed) {
   double energy = 0.0f;
   float thisnfp = targetRecipConn->fPatchSize();
   if( targetRecipConn->getReciprocalWgts() == NULL ) {
      targetRecipConn->setReciprocalWgts(targetRecipConn->getReciprocalWgtsName());
   }
   float recipnfp = targetRecipConn->getReciprocalWgts()->fPatchSize();
   for( int arbor=0; arbor<targetRecipConn->numberOfAxonalArborLists(); arbor++) {
      for( int k=0; k<targetRecipConn->getNumDataPatches(); k++) {
         PVPatch * p = targetRecipConn->getWeights(k, arbor);
         pvwdata_t * wdata = targetRecipConn->get_wDataHead(arbor, k);
         short int nx = p->nx;
         short int ny = p->ny;
         for( int n=0; n<nx*ny*targetRecipConn->fPatchSize(); n++ ) {
            int f = featureIndex(n,nx,ny,targetRecipConn->fPatchSize());
            const pvwdata_t * recipwdata = targetRecipConn->getReciprocalWgts()->get_wDataHead(arbor, f);
            //TODO-CER-2014.4.4 - convert weights here
            double wgtdiff = (wdata[n]/thisnfp - recipwdata[k]/recipnfp);
            energy += wgtdiff*wgtdiff;
         }
      }
   }
   energy *= targetRecipConn->getReciprocalFidelityCoeff()*0.5;
   return energy;
}

ReciprocalEnergyProbe::~ReciprocalEnergyProbe() {
}

} /* namespace PV */
