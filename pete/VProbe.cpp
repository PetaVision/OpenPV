/*
 * VProbe.cpp
 *
 *  Created on: Dec 17, 2010
 *      Author: pschultz
 */

#include "VProbe.hpp"

namespace PV {

VProbe::VProbe() : LayerProbe() {
}  // end ChannelProbe::ChannelProbe(ChannelType)

VProbe::VProbe(const char * filename) : LayerProbe(filename){
}  // end ChannelProbe::ChannelProbe(const char *, ChannelType)

int VProbe::outputState(float time, HyPerLayer * l) {
    pvdata_t * buf = l->getV();
    int n = l->getNumNeurons();
    fprintf(fp, "Layer %s at time %f:\n", l->getName(), time);
    for( int k=0; k<n; k++) {
        fprintf(fp, "    neuron %8d, value=%g\n", k, buf[k]);
    }
    return EXIT_SUCCESS;
}  // end ChannelProbe::outputState(float, HyPerLayer *)

}  // end namespace PV
