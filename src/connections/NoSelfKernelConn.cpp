/*
 * NoSelfKernelConn.cpp
 *
 *  Created on: Sep 20, 2011
 *      Author: gkenyon
 */

#include "NoSelfKernelConn.hpp"

namespace PV {

NoSelfKernelConn::NoSelfKernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
            ChannelType channel, const char * filename) : KernelConn(name, hc, pre, post,
                  channel, filename){};
NoSelfKernelConn::NoSelfKernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
            ChannelType channel, const char * filename, InitWeights *weightInit) : KernelConn(name, hc, pre, post,
                  channel, filename, weightInit){};
NoSelfKernelConn::NoSelfKernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
            ChannelType channel) : KernelConn(name, hc, pre, post,
                  channel){};

int NoSelfKernelConn::normalizeWeights(PVPatch ** patches, int numPatches, int arborId)
{
   //int axonID = 0;
   // self-interactions only defined for layers of same size
   assert(this->getPre()->getCLayer()->loc.nx == this->getPost()->getCLayer()->loc.nx);
   assert(this->getPre()->getCLayer()->loc.ny == this->getPost()->getCLayer()->loc.ny);
   assert(this->getPre()->getCLayer()->loc.nf == this->getPost()->getCLayer()->loc.nf);
   int num_kernels = this->numDataPatches();

   for (int kPatch = 0; kPatch < num_kernels; kPatch++) {
      PVPatch * wp = getKernelPatch(arborId, kPatch);
      pvdata_t * w = wp->data;
      int kfSelf = kPatch;
      int kxSelf = (nxp / 2);
      int kySelf = (nyp / 2);
      int kSelf = kIndex(kxSelf, kySelf, kfSelf, nxp, nyp, nfp);
      w[kSelf] = 0.0f;
   }
   return KernelConn::normalizeWeights(patches, numPatches, arborId);
}



} /* namespace PV */
