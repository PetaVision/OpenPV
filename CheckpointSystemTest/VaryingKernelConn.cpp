/*
 * VaryingKernelConn.cpp
 *
 *  Created on: Nov 10, 2011
 *      Author: pschultz
 */

#include "VaryingKernelConn.hpp"

namespace PV {

VaryingKernelConn::VaryingKernelConn(const char * name, HyPerCol * hc) : KernelConn() {
   initialize(name, hc);
}

VaryingKernelConn::~VaryingKernelConn() {}

int VaryingKernelConn::initialize(const char * name, HyPerCol * hc) {
   return KernelConn::initialize(name, hc);
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

int VaryingKernelConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag)
{
   KernelConn::ioParamsFillGroup(ioFlag);

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


