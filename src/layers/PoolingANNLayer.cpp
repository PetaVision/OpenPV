/*
 * PoolingANNLayer.cpp
 *
 * The output V is determined from GSynExc and GSynInh
 * using the formula GSynExc*GSynInh*(biasExc*GSynExc+biasInh*GSynInh)
 * biasExc and biasInh are set by the params file parameter bias:
 * biasExc = (1+bias)/2;  biasInh = (1-bias)/2
 *
 * This type of expression arises in the pooling generative models
 * "Exc" and "Inh" are really misnomers for this class, but the
 * terminology is inherited from the base class.
 *
 *  Created on: Apr 20, 2011
 *      Author: peteschultz
 */

#include "PoolingANNLayer.hpp"

namespace PV {

PoolingANNLayer::PoolingANNLayer() {
   initialize_base();
}

PoolingANNLayer::PoolingANNLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

int PoolingANNLayer::initialize_base() {
   return PV_SUCCESS;
}

int PoolingANNLayer::initialize(const char * name, HyPerCol * hc) {
   ANNLayer::initialize(name, hc, 2);
   PVParams * params = parent->parameters();
   setBias((pvdata_t) params->value(name, "bias", 0.0));
   return PV_SUCCESS;
}  // end of PoolingANNLayer::initialize()

int PoolingANNLayer::updateState(float timef, float dt) {
   return updateState(timef, dt, getNumNeurons(), getV(), getChannel(CHANNEL_EXC), getChannel(CHANNEL_INH), getBiasa(), getBiasb());
}

int PoolingANNLayer::updateState(float timef, float dt, int numNeurons, pvdata_t * V, pvdata_t * GSynExc, pvdata_t * GSynInh, pvdata_t biasa, pvdata_t biasb) {
   updateV_PoolingANNLayer(numNeurons, V, GSynExc, GSynInh, biasa, biasb);
   setActivity();
   resetGSynBuffers();
   updateActiveIndices();
   return PV_SUCCESS;
}

//int PoolingANNLayer::updateV() {
//   pvdata_t * V = getV();
//   pvdata_t * GSynExc = this->getChannel(CHANNEL_EXC);
//   pvdata_t * GSynInh = this->getChannel(CHANNEL_INH);
//   for( int k=0; k<getNumNeurons(); k++ ) {
//      V[k] = GSynExc[k]*GSynInh[k]*(getBiasa()*GSynExc[k]+getBiasb()*GSynInh[k]);
//   }
//   return PV_SUCCESS;
//}  // end of PoolingANNLayer::updateV()

}  // end of namespace PV block
