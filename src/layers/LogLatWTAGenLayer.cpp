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

int LogLatWTAGenLayer::initialize(const char * name, HyPerCol * hc) {
   GenerativeLayer::initialize(name, hc);
   sparsitytermcoeff = (pvdata_t) parent->parameters()->value(getName(),"sparsityTermCoefficient",1.0f);
   return PV_SUCCESS;
}

int LogLatWTAGenLayer::updateSparsityTermDerivative() {
   pvdata_t * V = getV();
   int nf = getLayerLoc()->nf;
   for( int k=0; k<getNumNeurons(); k+=nf) {
      // Assumes that stride in features is one.
      pvdata_t sumacrossfeatures = 0;
      for( int f=0; f<nf; f++ ) {
         sumacrossfeatures += V[k+f];
      }
      pvdata_t latWTAexpr = latWTAterm(V+k,nf); // a'*Lslash*a
      for( int f=0; f<nf; f++ ) {
         sparsitytermderivative[k+f] = 2*sparsitytermcoeff*(sumacrossfeatures-V[k+f])/(1+latWTAexpr);
      }
   }
   return PV_SUCCESS;
}  // end of LogLatWTAGenLayer::updateSparsityTermDerivative()

pvdata_t LogLatWTAGenLayer::latWTAterm(pvdata_t * V, int nf) {
   pvdata_t z=0;
   for( int p=0; p<nf; p++) {
      for( int q=0; q<nf; q++) {
         if( p!=q ) z += V[p]*V[q];
      }
   }
    return z;
}  // end of LogLatWTAGenLayer::latWTAterm(pvdata_t *, int)

}  // end of namespace PV block
