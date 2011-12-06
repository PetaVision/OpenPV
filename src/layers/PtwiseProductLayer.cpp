/*
 * PtwiseProductLayer.cpp
 *
 * The output V is the pointwise product of GSynExc and GSynInh
 *
 * "Exc" and "Inh" are really misnomers for this class, but the
 * terminology is inherited from the base class.
 *
 *  Created on: Apr 25, 2011
 *      Author: peteschultz
 */

#include "PtwiseProductLayer.hpp"

namespace PV {

PtwiseProductLayer::PtwiseProductLayer() {
   initialize_base();
}

PtwiseProductLayer::PtwiseProductLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}  // end PtwiseProductLayer::PtwiseProductLayer(const char *, HyPerCol *)

PtwiseProductLayer::~PtwiseProductLayer() {
}

int PtwiseProductLayer::initialize_base() {
   return PV_SUCCESS;
}

int PtwiseProductLayer::initialize(const char * name, HyPerCol * hc) {
   return ANNLayer::initialize(name, hc, 2);
}

int PtwiseProductLayer::updateV() {
    pvdata_t * V = getV();
    pvdata_t * GSynExc = getChannel(CHANNEL_EXC);
    pvdata_t * GSynInh = getChannel(CHANNEL_INH);
    for( int k=0; k<getNumNeurons(); k++ ) {
        V[k] = GSynExc[k] * GSynInh[k];
    }
    return PV_SUCCESS;
}  // end PtwiseProductLayer::updateV()

}  // end namespace PV
