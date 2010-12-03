/*
 * LateralConn.hpp
 *
 *  Created on: Nov 25, 2010
 *      Author: pschultz
 */

#ifndef LATERALCONN_HPP_
#define LATERALCONN_HPP_

#include "GenerativeConn.hpp"
#include <assert.h>
#include <string.h>

namespace PV {

class LateralConn : public GenerativeConn {
public:
    LateralConn();
    LateralConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, int channel);
    int initialize_base();
    int initialize(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, int channel);
    int initialize(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, int channel, const char * filename);
    int updateWeights(int axonID);

protected:
    PVPatch ** initializeWeights(PVPatch ** patches, int numPatches,
          const char * filename);
};

}  // end of block for namespace PV

#endif /* LATERALCONN_HPP_ */
