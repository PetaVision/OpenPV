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

int LogLatWTAGenLayer::updateState(float timef, float dt) {
   return updateState(timef, dt, getNumNeurons(), getV(), getChannel(CHANNEL_EXC), getChannel(CHANNEL_INH), getChannel(CHANNEL_INHB), sparsitytermderivative, dAold, VMax, VMin, VThresh, relaxation, auxChannelCoeff, sparsityTermCoeff, persistence);
}

int LogLatWTAGenLayer::updateState(float timef, float dt, int numNeurons, pvdata_t * V, pvdata_t * GSynExc, pvdata_t * GSynInh, pvdata_t * GSynAux, pvdata_t * sparsitytermderivative, pvdata_t * dAold, pvdata_t VMax, pvdata_t VMin, pvdata_t VThresh, pvdata_t relaxation, pvdata_t auxChannelCoeff, pvdata_t sparsityTermCoeff, pvdata_t persistence) {
   updateSparsityTermDeriv_LogLatWTAGenLayer(numNeurons, getLayerLoc()->nf, V, sparsitytermderivative);
   updateV_GenerativeLayer(numNeurons, V, GSynExc, GSynInh, GSynAux, sparsitytermderivative, dAold, VMax, VMin, VThresh, relaxation, auxChannelCoeff, sparsityTermCoeff, persistence);
   setActivity();
   resetGSynBuffers();
   updateActiveIndices();
   return PV_SUCCESS;
}

//int LogLatWTAGenLayer::updateSparsityTermDerivative() {
//   pvdata_t * V = getV();
//   int nf = getLayerLoc()->nf;
//   for( int k=0; k<getNumNeurons(); k+=nf) {
//      // Assumes that stride in features is one.
//      pvdata_t sumacrossfeatures = 0;
//      for( int f=0; f<nf; f++ ) {
//         sumacrossfeatures += V[k+f];
//      }
//      pvdata_t latWTAexpr = latWTAterm(V+k,nf); // a'*Lslash*a
//      for( int f=0; f<nf; f++ ) {
//         sparsitytermderivative[k+f] = 2*(sumacrossfeatures-V[k+f])/(1+latWTAexpr);
//      }
//   }
//   return PV_SUCCESS;
//}  // end of LogLatWTAGenLayer::updateSparsityTermDerivative()

//pvdata_t LogLatWTAGenLayer::latWTAterm(pvdata_t * V, int nf) {
//   pvdata_t z=0;
//   for( int p=0; p<nf; p++) {
//      for( int q=0; q<nf; q++) {
//         if( p!=q ) z += V[p]*V[q];
//      }
//   }
//    return z;
//}  // end of LogLatWTAGenLayer::latWTAterm(pvdata_t *, int)

}  // end of namespace PV block
