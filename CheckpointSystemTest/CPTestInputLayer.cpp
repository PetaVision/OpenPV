/*
 * CPTestInputLayer.cpp
 *
 *  Created on: Nov 10, 2011
 *      Author: pschultz
 */

#include "CPTestInputLayer.hpp"

namespace PV {

CPTestInputLayer::CPTestInputLayer(const char * name, HyPerCol * hc) : ANNLayer(name, hc) {
   initialize();
}

CPTestInputLayer::~CPTestInputLayer() {
}

int CPTestInputLayer::initialize() {
   initializeV(false);
   return PV_SUCCESS;
}

int CPTestInputLayer::initializeV(bool restart_flag) {
   // If restart_flag is true, initialize() will set V by calling readState()
   const PVLayerLoc * loc = getLayerLoc();
   if( !restart_flag ) {
      for (int k = 0; k < getNumNeurons(); k++){
         int kx = kxPos(k,loc->nx,loc->nx,loc->nf);
         int ky = kyPos(k,loc->nx,loc->ny,loc->nf);
         int kf = featureIndex(k,loc->nx,loc->ny,loc->nf);
         int kGlobal = kIndex(loc->kx0+kx,loc->ky0+ky,kf,loc->nxGlobal,loc->nyGlobal,loc->nf);
         getV()[k] = (pvdata_t) kGlobal;
      }
   }
   return PV_SUCCESS;
}

int CPTestInputLayer::updateV() {
   for( int k = 0; k < getNumNeurons(); k++ ) {
      getV()[k] += 1;
   }
   return PV_SUCCESS;
}

}  // end of namespace PV block


