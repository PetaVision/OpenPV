/*
 * PeriodicUpdateConn.cpp
 *
 *  Created on: May 18, 2011
 *      Author: peteschultz
 */

#include "PeriodicUpdateConn.hpp"

namespace PV {

PeriodicUpdateConn::PeriodicUpdateConn() {
    initialize_base();
}  // end of PeriodicUpdateConn::PeriodicUpdateConn()

PeriodicUpdateConn::PeriodicUpdateConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, ChannelType channel, const char * filename) {
   initialize_base();
   initialize(name, hc, pre, post, channel, filename );
}

PeriodicUpdateConn::~PeriodicUpdateConn() {
}

int PeriodicUpdateConn::initialize_base() {
   weightUpdatePeriod = 1.0;
   nextWeightUpdate = weightUpdatePeriod;
   return PV_SUCCESS;
}

int PeriodicUpdateConn::initialize( const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, ChannelType channel, const char * filename ) {
   PVParams * params = hc->parameters();
   weightUpdatePeriod = params->value(name, "weightUpdatePeriod", 1.0f);
   nextWeightUpdate = weightUpdatePeriod;
   KernelConn::initialize(name, hc, pre, post, channel, filename);
   return PV_SUCCESS;
}

int PeriodicUpdateConn::updateState(float time, float dt) {
    int status = PV_SUCCESS;
    if(time > nextWeightUpdate) {
        nextWeightUpdate += weightUpdatePeriod;
        status = updateWeights(0);
    }
    return status;
}  // end of PeriodicUpdateConn::updateState(float, float)

int PeriodicUpdateConn::updateWeights(int axonID) {
   printf("Call made at time %f to PeriodicUpdateConn::updateWeights(int) with argument axonID=%d\n",parent->simulationTime(), axonID);
   return PV_SUCCESS;
}

}  // end namespace PV
