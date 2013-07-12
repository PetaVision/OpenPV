/*
 * VaryingKernelConn.cpp
 *
 *  Created on: Nov 10, 2011
 *      Author: pschultz
 */

#include "VaryingKernelConn.hpp"

namespace PV {

VaryingKernelConn::VaryingKernelConn(const char * name, HyPerCol * hc,
      const char * pre_layer_name, const char * post_layer_name,
      const char * filename, InitWeights *weightInit) : KernelConn() {
   initialize(name, hc, pre_layer_name, post_layer_name, filename, weightInit);
}

VaryingKernelConn::~VaryingKernelConn() {}

int VaryingKernelConn::initialize(const char * name, HyPerCol * hc,
      const char * pre_layer_name, const char * post_layer_name,
      const char * filename, InitWeights *weightInit) {
   return KernelConn::initialize(name, hc, pre_layer_name, post_layer_name, filename, weightInit);
}

int VaryingKernelConn::allocateDataStructures() {
   KernelConn::allocateDataStructures();
   // initialize all dW's to one.
   int syPatch = yPatchStride();
   for(int kAxon = 0; kAxon < this->numberOfAxonalArborLists(); kAxon++){
      for(int kKernel = 0; kKernel < this->getNumDataPatches(); kKernel++){
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
   KernelConn::setParams(inputParams);

   return 0;
}

void VaryingKernelConn::readPlasticityFlag(PVParams * params) {
   plasticityFlag = true;
}

void VaryingKernelConn::readShmget_flag(PVParams * params) {
#ifdef USE_SHMGET
   shmget_flag = false;
#endif // USE_SHMGET
}

}  // end of namespace PV block


