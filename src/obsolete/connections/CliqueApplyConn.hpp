/*
 * CliqueApplyConn.hpp
 *
 *  Created on: Sep 8, 2011
 *      Author: gkenyon
 */

#ifndef CLIQUEAPPLYCONN_HPP_
#define CLIQUEAPPLYCONN_HPP_

#include "NoSelfKernelConn.hpp"

namespace PV {

class CliqueApplyConn: public NoSelfKernelConn {
public:

   CliqueApplyConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
               ChannelType channel, const char * filename, InitWeights *weightInit);
   int initialize(const char * name, HyPerCol * hc,
                  HyPerLayer * pre, HyPerLayer * post,
                  ChannelType channel, const char * filename,
                  InitWeights *weightInit=NULL);
   virtual int normalizeWeights(PVPatch ** patches, int numPatches, int arborId);

}; // class CliqueApplyConn

} /* namespace PV */
#endif /* CLIQUEAPPLYCONN_HPP_ */
