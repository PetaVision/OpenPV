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
    ChannelProbe(ChannelType channel);
    ChannelProbe(const char * filename, HyPerCol * hc, ChannelType channel);

    virtual int outputState(float time, HyPerLayer * l);
protected:
    ChannelType pChannel;

}; // end of class ChannelProbe


}  // end of namespace PV

#endif /* CHANNELPROBE_HPP_ */
