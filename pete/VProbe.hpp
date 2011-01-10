/*
 * VProbe.hpp
 *
 *  Created on: Dec 17, 2010
 *      Author: pschultz
 */

#ifndef VPROBE_HPP_
#define VPROBE_HPP_

#include "../PetaVision/src/io/LayerProbe.hpp"
#include "../PetaVision/src/include/pv_types.h"
#include "../PetaVision/src/layers/HyPerLayer.hpp"

namespace PV {

class VProbe : public LayerProbe {
public:
    VProbe();
    VProbe(const char * filename);

    virtual int outputState(float time, HyPerLayer * l);

}; // end of class ChannelProbe

}  // end of namespace PV

#endif /* VPROBE_HPP_ */
