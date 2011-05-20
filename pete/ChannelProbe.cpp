/*
 * ChannelProbe.cpp
 *
 *  Created on: Nov 22, 2010
 *      Author: pschultz
 */

#include "ChannelProbe.hpp"

namespace PV {

ChannelProbe::ChannelProbe(ChannelType channel) : LayerProbe() {
    pChannel = channel;
}  // end ChannelProbe::ChannelProbe(ChannelType)

ChannelProbe::ChannelProbe(const char * filename, HyPerCol * hc, ChannelType channel) : LayerProbe(filename, hc){
    pChannel = channel;
}  // end ChannelProbe::ChannelProbe(const char *, HyPerCol *, ChannelType)

int ChannelProbe::outputState(float time, HyPerLayer * l) {
    pvdata_t * buf = l->getChannel(pChannel);
    int n = l->getNumNeurons();
    for( int k=0; k<n; k++) {
        fprintf(fp, "Layer %s, channel %d, time %f, neuron %8d, value=%10.8f\n",
        		l->getName(), (int) pChannel, time, k, buf[k]);
    }
    return EXIT_SUCCESS;
}  // end ChannelProbe::outputState(float, HyPerLayer *)

}  // end namespace PV
