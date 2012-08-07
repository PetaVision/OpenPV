/*
 * VaryingHyPerConn.cpp
 *
 *  Created on: Nov 10, 2011
 *      Author: pschultz
 */

#include "VaryingHyPerConn.hpp"

namespace PV {

VaryingHyPerConn::VaryingHyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
      const char * filename, InitWeights *weightInit) : HyPerConn() {
   initialize(name, hc, pre, post, filename, weightInit);
}

VaryingHyPerConn::~VaryingHyPerConn() {}

int VaryingHyPerConn::initialize(const char * name, HyPerCol * hc,
         HyPerLayer * pre, HyPerLayer * post,
         const char * filename, InitWeights *weightInit) {
   HyPerConn::initialize(name, hc, pre, post, filename, weightInit);

   // initialize all dW's to one.
   int syPatch = yPatchStride();
   for(int kAxon = 0; kAxon < numberOfAxonalArborLists(); kAxon++){
      for(int kPatch = 0; kPatch < getNumDataPatches(); kPatch++){
         PVPatch * W = getWeights(kPatch, kAxon);
         int nkPatch = fPatchSize() * W->nx;
         float * dWdata = get_dwData(kAxon, kPatch);
         for(int kyPatch = 0; kyPatch < W->ny; kyPatch++){
            for(int kPatch = 0; kPatch < nkPatch; kPatch++){
               dWdata[kPatch] = 1.0f;
            }
            dWdata += syPatch;
         }
      }
   }
   return PV_SUCCESS;
}

int VaryingHyPerConn::calc_dW(int axonId) {
   // keep all dW's at one.
   return PV_SUCCESS;
}

int VaryingHyPerConn::updateWeights(int axonId) {
   int syPatch = yPatchStride();
   for( int kPatch = 0; kPatch < getNumDataPatches(); kPatch++) {
      PVPatch * W = getWeights(kPatch, axonId);
      int nkPatch = fPatchSize() * W->nx;
      float * Wdata = get_wData(axonId, kPatch); // W->data;
      pvdata_t * dWdata = get_dwData(axonId, kPatch);
      for(int kyPatch = 0; kyPatch < W->ny; kyPatch++) {
         for(int kPatch = 0; kPatch < nkPatch; kPatch++) {
            Wdata[kPatch] += dWdata[kPatch];
         }
         dWdata += syPatch;
      }
   }
   return PV_SUCCESS;
}

int VaryingHyPerConn::setParams(PVParams * inputParams)
{
   const char * name = getName();

   numAxonalArborLists=(int) inputParams->value(name, "numAxonalArbors", 1, true);
   plasticityFlag = true;
   stochasticReleaseFlag = inputParams->value(name, "stochasticReleaseFlag", false, true) != 0;

   writeCompressedWeights = inputParams->value(name, "writeCompressedWeights", true);

   return 0;
}

}  // end of namespace PV block


