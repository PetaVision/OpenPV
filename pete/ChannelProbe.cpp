/*
 * ChannelProbe.cpp
 *
 *  Created on: Nov 22, 2010
 *      Author: pschultz
 */

#include "ChannelProbe.hpp"

namespace PV {

ChannelProbe::ChannelProbe(HyPerLayer * layer, ChannelType channel) : LayerProbe() {
   initLayerProbe(NULL, layer);
   pChannel = channel;
}  // end ChannelProbe::ChannelProbe(ChannelType)

ChannelProbe::ChannelProbe(const char * filename, HyPerLayer * layer, ChannelType channel) : LayerProbe(){
   initLayerProbe(filename, layer);
   pChannel = channel;
}  // end ChannelProbe::ChannelProbe(const char *, HyPerCol *, ChannelType)

int ChannelProbe::outputState(float timef) {
    pvdata_t * buf = getTargetLayer()->getChannel(pChannel);
    int n = getTargetLayer()->getNumNeurons();
    for( int k=0; k<n; k++) {
        fprintf(fp, "Layer %s, channel %d, time %f, neuron %8d, value=%.8g\n",
        		getTargetLayer()->getName(), (int) pChannel, timef, k, buf[k]);
    }
    return EXIT_SUCCESS;
}  // end ChannelProbe::outputState(float, HyPerLayer *)

}  // end namespace PV
