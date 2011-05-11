/*
 * IdentConn.hpp
 *
 *  Created on: Nov 17, 2010
 *      Author: pschultz
 */

#ifndef IDENTCONN_HPP_
#define IDENTCONN_HPP_

#include "KernelConn.hpp"
#include <assert.h>
#include <string.h>

namespace PV {

class IdentConn : public KernelConn {
public:
    IdentConn();
    IdentConn(const char * name, HyPerCol *hc, HyPerLayer * pre, HyPerLayer * post, ChannelType channel);
    int initialize_base();
    virtual int updateWeights(int axonID) {return PV_SUCCESS;}

protected:
    int setPatchSize(const char * filename);
    virtual PVPatch ** initializeWeights(PVPatch ** patches, int numPatches,
          const char * filename);
};

}  // end of block for namespace PV


#endif /* IDENTCONN_HPP_ */
