/*
 * NoSelfKernelConn.cpp
 *
 *  Created on: Sep 20, 2011
 *      Author: gkenyon
 */

#include "NoSelfKernelConn.hpp"

namespace PV {

NoSelfKernelConn::NoSelfKernelConn()
{
}

NoSelfKernelConn::NoSelfKernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
            ChannelType channel, const char * filename, InitWeights *weightInit) {
   KernelConn::initialize(name, hc, pre, post,channel, filename, weightInit);
};

int NoSelfKernelConn::zeroSelfWeights(PVPatch ** patches, int numPatches, int arborId){
   //int axonID = 0;
   // self-interactions only defined for layers of same size
   assert(this->getPre()->getCLayer()->loc.nx == this->getPost()->getCLayer()->loc.nx);
   assert(this->getPre()->getCLayer()->loc.ny == this->getPost()->getCLayer()->loc.ny);
   assert(this->getPre()->getCLayer()->loc.nf == this->getPost()->getCLayer()->loc.nf);
   int num_kernels = this->numDataPatches();

   // because the default return value/behavior of KernelConn::normalizeWeights is PV_BREAK,
   // safest approach here is to zero self-interactions for all arbors
   assert(arborId == 0);  // necessary?  could execute this routine numAxonArbors times without apparent harm
   for (int axonIndex = 0; axonIndex < this->numberOfAxonalArborLists(); axonIndex++) {
      for (int kPatch = 0; kPatch < num_kernels; kPatch++) {
         // PVPatch * wp = getWeights(axonIndex, kPatch); // getKernelPatch(axonIndex, kPatch);
         pvdata_t * w = get_wData(axonIndex, kPatch); // wp->data;
         int kfSelf = kPatch;
         int kxSelf = (nxp / 2);
         int kySelf = (nyp / 2);
         int kSelf = kIndex(kxSelf, kySelf, kfSelf, nxp, nyp, nfp);
         w[kSelf] = 0.0f;
      } // kPatch
   }  // axonIndex
   return PV_BREAK;
}

int NoSelfKernelConn::normalizeWeights(PVPatch ** patches, pvdata_t ** dataStart, int numPatches, int arborId)
{
   int status = zeroSelfWeights(patches, numPatches, arborId);
   assert( (status == PV_SUCCESS) || (status == PV_BREAK) );
   return KernelConn::normalizeWeights(patches, dataStart, numPatches, arborId);  // parent class should return PV_BREAK
}



} /* namespace PV */
