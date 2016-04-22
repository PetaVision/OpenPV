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

NoSelfKernelConn::NoSelfKernelConn(const char * name, HyPerCol * hc) {
   HyPerConn::initialize(name, hc);
}

// TODO: make sure code works in non-shared weight case
void NoSelfKernelConn::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   sharedWeights = true;
   if (ioFlag == PARAMS_IO_READ) {
      fileType = PVP_KERNEL_FILE_TYPE;
      parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights", true/*correctValue*/);
   }
}

int NoSelfKernelConn::normalizeWeights() {
   zeroSelfWeights(getNumDataPatches(), 0);
   return HyPerConn::normalizeWeights();
}

int NoSelfKernelConn::zeroSelfWeights(int numPatches, int arborId){
   //int axonID = 0;
   // self-interactions only defined for layers of same size
   assert(this->getPre()->getCLayer()->loc.nx == this->getPost()->getCLayer()->loc.nx);
   assert(this->getPre()->getCLayer()->loc.ny == this->getPost()->getCLayer()->loc.ny);
   assert(this->getPre()->getCLayer()->loc.nf == this->getPost()->getCLayer()->loc.nf);
   int num_kernels = this->getNumDataPatches();

   // because the default return value/behavior of HyPerConn::normalizeWeights is PV_BREAK,
   // safest approach here is to zero self-interactions for all arbors
   assert(arborId == 0);  // necessary?  could execute this routine numAxonArbors times without apparent harm
   for (int axonIndex = 0; axonIndex < this->numberOfAxonalArborLists(); axonIndex++) {
      for (int kPatch = 0; kPatch < num_kernels; kPatch++) {
         // PVPatch * wp = getWeights(axonIndex, kPatch); // getKernelPatch(axonIndex, kPatch);
         pvwdata_t * w = get_wDataHead(axonIndex, kPatch); // wp->data;
         int kfSelf = kPatch;
         int kxSelf = (nxp / 2);
         int kySelf = (nyp / 2);
         int kSelf = kIndex(kxSelf, kySelf, kfSelf, nxp, nyp, nfp);
         w[kSelf] = 0.0f;
      } // kPatch
   }  // axonIndex
   return PV_BREAK;
}

} /* namespace PV */
