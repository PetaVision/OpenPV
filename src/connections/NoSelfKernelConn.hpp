/*
 * NoSelfKernelConn.hpp
 *
 *  Created on: Sep 20, 2011
 *      Author: gkenyon
 */

#ifndef NOSELFKERNELCONN_HPP_
#define NOSELFKERNELCONN_HPP_

#include "KernelConn.hpp"

namespace PV {

class NoSelfKernelConn: public PV::KernelConn {
public:
   NoSelfKernelConn();

   NoSelfKernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
               ChannelType channel, const char * filename, InitWeights *weightInit);
   int zeroSelfWeights(PVPatch ** patches, int numPatches, int arborId);
   virtual int normalizeWeights(PVPatch ** patches, int numPatches, int arborId);
};

} /* namespace PV */
#endif /* NOSELFKERNELCONN_HPP_ */
