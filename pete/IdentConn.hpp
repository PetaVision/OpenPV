/*
 * IdentConn.hpp
 *
 *  Created on: Nov 17, 2010
 *      Author: pschultz
 */

#ifndef IDENTCONN_HPP_
#define IDENTCONN_HPP_

#include "../PetaVision/src/connections/KernelConn.hpp"
#include <assert.h>
#include <string.h>

namespace PV {

class IdentConn : public KernelConn {
public:
    IdentConn(const char * name, HyPerCol *hc, HyPerLayer * pre, HyPerLayer * post, int channel);
    int initialize(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, int channel);
    int updateWeights(int axonID) {return EXIT_SUCCESS;}

protected:
    int setPatchSize(const char * filename);
    PVPatch ** initializeWeights(PVPatch ** patches, int numPatches,
          const char * filename);
};

}  // end of block for namespace PV


#endif /* IDENTCONN_HPP_ */
