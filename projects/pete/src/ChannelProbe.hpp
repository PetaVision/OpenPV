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
    ChannelProbe(const char * probeName, HyPerCol * hc);

    virtual int outputState(double timed);
protected:
    int initChannelProbe(const char * probeName, HyPerCol * hc);
    virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
    virtual void ioParam_channelCode(enum ParamsIOFlag ioFlag);

private:
    int initChannelProbe_base();

    ChannelType pChannel;

}; // end of class ChannelProbe


}  // end of namespace PV

#endif /* CHANNELPROBE_HPP_ */
