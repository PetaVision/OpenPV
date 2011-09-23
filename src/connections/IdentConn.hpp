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

class InitIdentWeights;

class IdentConn : public KernelConn {
public:
    IdentConn();
    IdentConn(const char * name, HyPerCol *hc,
            HyPerLayer * pre, HyPerLayer * post, ChannelType channel, InitWeights *weightInitializer);

   virtual int initialize_base();
   virtual int initialize(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post, ChannelType channel, const char * filename, InitWeights *weightInit);
   virtual int setParams(PVParams * inputParams);
   virtual int updateWeights(int axonID) {return PV_SUCCESS;}
   virtual int initNormalize();

protected:
    int setPatchSize(const char * filename);
    //virtual PVPatch ** initializeWeights(PVPatch ** patches, int numPatches,
    //      const char * filename);
};

}  // end of block for namespace PV


#endif /* IDENTCONN_HPP_ */
