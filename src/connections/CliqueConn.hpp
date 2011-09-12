/*
 * CliqueConn.hpp
 *
 *  Created on: Sep 8, 2011
 *      Author: gkenyon
 */

#ifndef CLIQUECONN_HPP_
#define CLIQUECONN_HPP_

#include "KernelConn.hpp"

namespace PV {

class CliqueConn: public KernelConn {
public:

   CliqueConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
               ChannelType channel, const char * filename);
   CliqueConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
               ChannelType channel, const char * filename, InitWeights *weightInit);
   CliqueConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
               ChannelType channel);
   virtual int calc_dW(int axonId);
   virtual int updateState(float time, float dt);
   virtual int updateWeights(int arbor);
   virtual PVPatch ** normalizeWeights(PVPatch ** patches, int numPatches, int arborId);

}; // class CliqueConn

} /* namespace PV */
#endif /* CLIQUECONN_HPP_ */
