/*
 * LateralGenConn.hpp
 *
 *  Created on: Nov 25, 2010
 *      Author: pschultz
 */

#ifndef LATERALGENCONN_HPP_
#define LATERALGENCONN_HPP_

#include "../PetaVision/src/connections/GenerativeConn.hpp"
#include <string.h>

namespace PV {

class LateralGenConn : public GenerativeConn {
public:
    LateralGenConn();
    LateralGenConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, ChannelType channel);
    LateralGenConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, ChannelType channel, const char * filename);
    int initialize_base();
    int initialize(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, ChannelType channel);
    int initialize(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, ChannelType channel, const char * filename);
    int updateWeights(int axonID);

protected:
    PVPatch *** initializeWeights(PVPatch *** patches, pvdata_t ** dataStart,
          int numPatches, const char * filename);
};

}  // end of block for namespace PV

#endif /* LATERALGENCONN_HPP_ */
