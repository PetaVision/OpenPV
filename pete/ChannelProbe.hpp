/*
 * ChannelProbe.hpp
 *
 *  Created on: Nov 22, 2010
 *      Author: pschultz
 */

#ifndef CHANNELPROBE_HPP_
#define CHANNELPROBE_HPP_

#include "../PetaVision/src/io/LayerProbe.hpp"
#include "../PetaVision/src/include/pv_types.h"
#include "../PetaVision/src/layers/HyPerLayer.hpp"

namespace PV {

class ChannelProbe : public LayerProbe {
public:
    ChannelProbe(HyPerLayer * layer, ChannelType channel);
    ChannelProbe(const char * filename, HyPerLayer * layer, ChannelType channel);

    virtual int outputState(double timed);
protected:
    ChannelType pChannel;

}; // end of class ChannelProbe


}  // end of namespace PV

#endif /* CHANNELPROBE_HPP_ */
