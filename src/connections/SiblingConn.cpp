/*
 * SiblingConn.cpp
 *
 *  Created on: Jan 26, 2012
 *      Author: garkenyon
 */

#include "SiblingConn.hpp"

namespace PV {

SiblingConn::SiblingConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
      ChannelType channel, const char * filename, InitWeights *weightInit, SiblingConn *sibling_conn)
{
   SiblingConn::initialize_base();
   SiblingConn::initialize(name, hc, pre, post, channel, filename, weightInit, sibling_conn);
   // HyPerConn::initialize is not virtual
}

int SiblingConn::initialize(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, ChannelType channel, const char * filename,
      InitWeights *weightInit, SiblingConn *sibling_conn)
{
   siblingConn = sibling_conn;
   isNormalized = false;
   if (siblingConn != NULL){
      siblingConn->setSiblingConn(this);
   }
   return KernelConn::initialize(name, hc, pre, post, channel, filename, weightInit);
}

int SiblingConn::initNormalize(){
   KernelConn::initNormalize();
   isNormalized = true;
   return PV_BREAK;
}

bool SiblingConn::getIsNormalized(){
   return isNormalized;
}

void SiblingConn::setSiblingConn(SiblingConn *sibling_conn){
   assert((siblingConn) == NULL || (siblingConn == sibling_conn));
   siblingConn = sibling_conn;
}

int SiblingConn::normalizeFamily()
{
   // normalize all arbors individuqlly relative to siblings
   const int num_kernels = numDataPatches();
   for (int kArbor = 0; kArbor < this->numberOfAxonalArborLists(); kArbor++) {
      for (int kPatch = 0; kPatch < num_kernels; kPatch++) {
         // PVPatch * localWPatch = getWeights(kPatch,kArbor); // this->getKernelPatch(kArbor, kPatch);
         // pvdata_t * myWeights = myWpatch->data;
         pvdata_t * localWeights = get_wDataHead(kArbor, kPatch);
         assert(localWeights != NULL);
         // PVPatch * siblingWpatch = siblingConn->getKernelPatch(kArbor, kPatch);
         // pvdata_t * siblingWeights = siblingWpatch->data;
         pvdata_t * siblingWeights = siblingConn->get_wDataHead(kArbor, kPatch);
         assert(siblingWeights != NULL);
         const int nx = nxp; // localWPatch->nx;
         const int ny = nyp; // localWPatch->ny;
         const int nf = nfp;
         const int sy = syp;
         for (int ky = 0; ky < ny; ky++) {
            for (int iWeight = 0; iWeight < nf * nx; iWeight++) {
               pvdata_t norm_denom = fabs(siblingWeights[iWeight]) + fabs(localWeights[iWeight]);
               norm_denom = (norm_denom != 0.0f) ? norm_denom : 1.0f;
               localWeights[iWeight] /= norm_denom;
               siblingWeights[iWeight] /= norm_denom;
            }
            localWeights += sy;
            siblingWeights += sy;
         }
      } // kPatch < numPatches
   } // kArbor
   return PV_BREAK;
} // normalizeFamily

int SiblingConn::normalizeWeights(PVPatch ** patches, pvdata_t ** dataStart, int numPatches, int arborId)
{
   int status = PV_SUCCESS;

   // individually normalize each arbor for self and sibling
   if (this->numberOfAxonalArborLists() > 1) {
      assert(this->normalize_arbors_individually);
   }
   status = NoSelfKernelConn::normalizeWeights(patches, dataStart, numPatches, arborId);  // parent class should return PV_BREAK
   assert( (status == PV_SUCCESS) || (status == PV_BREAK) );

   if ((siblingConn != NULL) && (siblingConn->getIsNormalized())){
      status = this->normalizeFamily();
   }

   return PV_BREAK;
} // normalizeWeights


} // namespace PV
