/*
 * ChannelProbe.hpp
 *
 *  Created on: Nov 22, 2010
 *      Author: pschultz
 */

#ifndef CHANNELPROBE_HPP_
#define CHANNELPROBE_HPP_

#include <io/LayerProbe.hpp>
#include <include/pv_types.h>
#include <layers/HyPerLayer.hpp>

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
