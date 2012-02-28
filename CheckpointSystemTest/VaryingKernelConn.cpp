/*
 * VaryingKernelConn.cpp
 *
 *  Created on: Nov 10, 2011
 *      Author: pschultz
 */

#include "VaryingKernelConn.hpp"

namespace PV {

VaryingKernelConn::VaryingKernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
      ChannelType channel, const char * filename, InitWeights *weightInit) : KernelConn() {
   initialize(name, hc, pre, post, channel, filename, weightInit);
}

VaryingKernelConn::~VaryingKernelConn() {}

int VaryingKernelConn::initialize(const char * name, HyPerCol * hc,
         HyPerLayer * pre, HyPerLayer * post, ChannelType channel,
         const char * filename, InitWeights *weightInit) {
   KernelConn::initialize(name, hc, pre, post, channel, filename, weightInit);

   // initialize all dW's to one.
   int syPatch = yPatchStride();
   for(int kAxon = 0; kAxon < this->numberOfAxonalArborLists(); kAxon++){
      for(int kKernel = 0; kKernel < this->numDataPatches(); kKernel++){
         PVPatch * patch = getWeights(kKernel, kAxon); // dKernelPatches[kAxon][kKernel];
         int nkPatch = fPatchSize() * patch->nx;
         float * dWeights = get_dwData(kAxon, kKernel); // dKernelPatch->data;
         for(int kyPatch = 0; kyPatch < patch->ny; kyPatch++){
            for(int kPatch = 0; kPatch < nkPatch; kPatch++){
               dWeights[kPatch] = 1.0f;
            }
            dWeights += syPatch;
         }
      }
   }
   return PV_SUCCESS;
}

int VaryingKernelConn::calc_dW(int axonId) {
   // keep all dW's at one.
   return PV_SUCCESS;
}

int VaryingKernelConn::setParams(PVParams * inputParams /*, PVConnParams * p*/)
{
   const char * name = getName();

   numAxonalArborLists=(int) inputParams->value(name, "numAxonalArbors", 1, true);
   plasticityFlag = true;
   stochasticReleaseFlag = inputParams->value(name, "stochasticReleaseFlag", false, true) != 0;

   writeCompressedWeights = inputParams->value(name, "writeCompressedWeights", true);

   return 0;
}

}  // end of namespace PV block


