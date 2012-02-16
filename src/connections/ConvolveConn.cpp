/*
 * ConvolveConn.cpp
 *
 *  Created on: Oct 5, 2009
 *      Author: rasmussn
 */

#include "ConvolveConn.hpp"

namespace PV {

ConvolveConn::ConvolveConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
                           ChannelType channel, InitWeights *weightInit)
{
   HyPerConn::initialize(name, hc, pre, post, channel, NULL, weightInit);
}

int ConvolveConn::initialize(const char * filename)
{
   //
   // patch is in pre-synaptic layer
      // hum, maybe not, the patch is the kernel if you are applying weights
   patch.data = pre->clayer->V;
   patch.nx = nxp;
   patch.ny = nyp;
   //patch.nf = nfp;
   //patch.sx = nfp;
   //patch.sy = nfp * nxp;
   //patch.sf = 1;

   return 0;
}

int ConvolveConn::deliver(PVLayerCube * cube, int neighbor)
{
   convolve(cube, cube, &patch);
   return 0;
}

void ConvolveConn::convolve(PVLayerCube * dst, PVLayerCube * src, PVPatch * patch)
{
   // forall neurons in post-synaptic layer
      // get pre-synaptic patch head (e.g., every 8th post, shift one pre)
            // this works as if nf==8, interesting
            // could cycle through patches (Gar's kernelconn stuff) -> could have 8 patches
      // forall neurons in patch, apply convolution
      // store in post-synaptic layer
}

} // namespace PV
