/*
 * WindowLCALayera.cpp
 *
 *  Created on: Jan 24, 2013
 *      Author: garkenyon
 */

#include "WindowLCALayer.hpp"
#include <layers/ANNLayer.cpp>

namespace PV {

WindowLCALayer::WindowLCALayer(const char * name, HyPerCol * hc, int num_channels)
{
   HyPerLCALayer::initialize(name, hc, num_channels);
}

WindowLCALayer::WindowLCALayer(const char * name, HyPerCol * hc)
{
   HyPerLCALayer::initialize(name, hc, 2);
}

WindowLCALayer::~WindowLCALayer()
{
}

//Overwrite LCA to ANNLayer's updateState
int WindowLCALayer::doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
      pvdata_t * V, int num_channels, pvdata_t * gSynHead, bool spiking,
      unsigned int * active_indices, unsigned int * num_active){
   int status = ANNLayer::doUpdateState(time, dt, loc, A, V, num_channels, gSynHead, spiking, active_indices, num_active);
   return status;
}

}
