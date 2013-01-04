/*
 * MaxPooling.cpp
 *
 *  Created on: Jan 3, 2013
 *      Author: gkenyon
 */

#include "MaxPooling.hpp"

namespace PV {

MaxPooling::MaxPooling()
{
   initialize_base();
}

MaxPooling::MaxPooling(const char * name, HyPerCol * hc, int numChannels)
{
   initialize_base();
   initialize(name, hc, numChannels);
}

MaxPooling::MaxPooling(const char * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc, MAX_CHANNELS);
}

MaxPooling::~MaxPooling()
{
}

int MaxPooling::initialize_base(){
   return PV_SUCCESS;
}

int MaxPooling::initialize(const char * name, HyPerCol * hc, int numChannels)
{
   return ANNLayer::initialize(name, hc, numChannels);
}

int MaxPooling::recvSynapticInput(HyPerConn * conn, const PVLayerCube * activity,
      int axonId)
{
   return HyPerLayer::recvSynapticInput(conn, activity, axonId );
}

} // namespace PV
