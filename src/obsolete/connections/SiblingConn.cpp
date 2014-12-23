/*
 * SiblingConn.cpp
 *
 *  Created on: Jan 26, 2012
 *      Author: garkenyon
 */

#include "SiblingConn.hpp"

namespace PV {

SiblingConn::SiblingConn(const char * name, HyPerCol * hc) {
   SiblingConn::initialize_base();
   SiblingConn::initialize(name, hc);
   // HyPerConn::initialize is not virtual
}

int SiblingConn::initialize(const char * name, HyPerCol * hc) {
   isNormalized = false; // TODO: check that isNormalized is set and cleared properly.
   return HyPerConn::initialize(name, hc);
}

int SiblingConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   ioParam_siblingConnName(ioFlag);
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   return status;
}

void SiblingConn::ioParam_siblingConnName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "siblingConnName", &siblingConnName);
}

void SiblingConn::ioParam_normalizeMethod(enum ParamsIOFlag ioFlag) {
   HyPerConn::ioParam_normalizeMethod(ioFlag);
   isNormalized = true; // TODO: check that isNormalized is set and cleared properly.
}

int SiblingConn::communicateInitInfo() {
   int status = NoSelfKernelConn::communicateInitInfo();
   HyPerConn * hyper_conn = parent->getConnFromName(siblingConnName);
   siblingConn = dynamic_cast<SiblingConn *>(hyper_conn);
   if (siblingConn != NULL) {
      siblingConn->setSiblingConn(this);
   }
   return status;
}

bool SiblingConn::getIsNormalized() {
   return isNormalized;
}

void SiblingConn::setSiblingConn(SiblingConn *sibling_conn) {
   assert((siblingConn) == NULL || (siblingConn == sibling_conn));
   siblingConn = sibling_conn;
}

int SiblingConn::normalizeFamily() {
   // normalize all arbors individuqlly relative to siblings
   // insert MPI_Barrier before callng SiblingConn::normalizeFamily to ensure all processors have finished self normalization
   const int num_kernels = getNumDataPatches();

   // first scale each weight by sum of the absolute values of the local weight plus the sibling weight
#ifdef OBSOLETE // Marked obsolete Dec 9, 2014.
   // if using shared memory (shmget) then the
   // process that owns the local weight scales the sibling weight as well, even if not the owner of the sibling weight
#endif // OBSOLETE
   for (int kArbor = 0; kArbor < this->numberOfAxonalArborLists(); kArbor++) {
      for (int kPatch = 0; kPatch < num_kernels; kPatch++) {
         pvwdata_t * localWeights = get_wDataHead(kArbor, kPatch);
         assert(localWeights != NULL);
#ifdef OBSOLETE // Marked obsolete Dec 9, 2014.
#ifdef USE_SHMGET
         volatile pvwdata_t * siblingWeights = siblingConn->get_wDataHead(
               kArbor, kPatch);
#else
         pvwdata_t * siblingWeights = siblingConn->get_wDataHead(kArbor, kPatch);
#endif // USE_SHMGET
#endif // OBSOLETE
         assert(siblingWeights != NULL);
         for (int iWeight = 0; iWeight < nxp * nyp * nfp; iWeight++) {
            pvdata_t norm_denom = fabs(siblingWeights[iWeight])
						      + fabs(localWeights[iWeight]);
            norm_denom = (norm_denom != 0.0f) ? norm_denom : 1.0f;
#ifdef USE_SHMGET
            if (shmget_flag) {
               if (shmget_owner[kArbor]) {
                  localWeights[iWeight] /= norm_denom;
               }
            } else {
               localWeights[iWeight] /= norm_denom;
            }
#else
            localWeights[iWeight] /= norm_denom;
#endif
#ifdef USE_SHMGET
            if (siblingConn->getShmgetFlag()) {  // sibling is using shared memory, only one process should adjust sibling weights
               if (shmget_flag){  // local conn is using shared memory, use it's owner flag
                  //if (siblingConn->getShmgetOwner(kArbor)) {  // won't work! owner of siblingConn may be different process
                  if (shmget_owner[kArbor]) {
                     siblingWeights[iWeight] /= norm_denom;
                  }
               }  // local conn is not using shared memory, use sibling's owner flag, ok because no other process can write to this conn
               else if (siblingConn->getShmgetOwner(kArbor)) {
                  siblingWeights[iWeight] /= norm_denom;
               }
            } else {  //sibling is not using shared memory, so each process adjusts sibling weights
               assert(!shmget_flag); // if local conn is shared, then local weight may have already been adjusted by another process and denom may not be correct
               siblingWeights[iWeight] /= norm_denom;
            }
#else
            siblingWeights[iWeight] /= norm_denom;
#endif
         } // iWeight
      } // kPatch < numPatches
   } // kArbor
#ifdef PV_USE_MPI
#ifdef USE_SHMGET
   // insert barrier to ensure that all processes have finished scaling weights before computing sums
   //	std::cout << "entering MPI_Barrier in SiblingConn::normalizeFamily: " << this->name << ", rank = " << getParent()->icCommunicator()->commRank() << std::endl;
   MPI_Barrier(parent->icCommunicator()->communicator());
   //	std::cout << "exiting MPI_Barrier in SiblingConn::normalizeFamily: " << this->name << ", rank = " << getParent()->icCommunicator()->commRank() << std::endl;
#endif
#endif

   // compute sums over local and sibling weights over all patches and arbors
   double sum_local = 0.0;
   double sum_sibling = 0.0;
   for (int kArbor = 0; kArbor < this->numberOfAxonalArborLists(); kArbor++) {
      for (int kPatch = 0; kPatch < num_kernels; kPatch++) {
         pvwdata_t * localWeights = get_wDataHead(kArbor, kPatch);
         assert(localWeights != NULL);
#ifdef USE_SHMGET
         volatile pvwdata_t * siblingWeights = siblingConn->get_wDataHead(
               kArbor, kPatch);
#else
         pvwdata_t * siblingWeights = siblingConn->get_wDataHead(kArbor, kPatch);
#endif
         assert(siblingWeights != NULL);
         for (int iWeight = 0; iWeight < nxp * nyp * nfp; iWeight++) {
            sum_local += localWeights[iWeight];
            sum_sibling += siblingWeights[iWeight];
         } // iWeight
      } // kPatch < numPatches
   } // kArbor

   // finally, scale local weights so that average local == average sibling
   float scale_factor =
         sum_local != 0 ? fabs(sum_sibling) / fabs(sum_local) : 1;
   for (int kArbor = 0; kArbor < this->numberOfAxonalArborLists(); kArbor++) {
#ifdef USE_SHMGET
      if (shmget_flag && !shmget_owner[kArbor]) {
         continue;
      }
      volatile pvwdata_t * localWeights = this->get_wDataStart(kArbor);
#else
      pvwdata_t * localWeights = this->get_wDataStart(kArbor);
#endif
      for (int iWeight = 0; iWeight < nxp * nyp * nfp; iWeight++) {
         localWeights[iWeight] *= scale_factor;
      }
   }
   return PV_BREAK;
} // normalizeFamily

int SiblingConn::normalizeWeights() { // (PVPatch ** patches, pvdata_t ** dataStart, int numPatches, int arborId) {
   // int status = PV_SUCCESS;

   int status = NoSelfKernelConn::normalizeWeights();
   // status = NoSelfKernelConn::normalizeWeights(patches, dataStart, numPatches, arborId);  // parent class should return PV_BREAK
   assert( (status == PV_SUCCESS) || (status == PV_BREAK));

   if ((siblingConn != NULL) && (siblingConn->getIsNormalized())) {
#ifdef PV_USE_MPI
#ifdef USE_SHMGET
      // insert barrier to ensure that all processes have finished individual normalization before starting family normalization
      //		std::cout << "entering MPI_Barrier in SiblingConn::normalizeWeights: " << this->name << ", rank = " << getParent()->icCommunicator()->commRank() << std::endl;
      MPI_Barrier(parent->icCommunicator()->communicator());
      //		std::cout << "exiting MPI_Barrier in SiblingConn::normalizeWeights: " << this->name << ", rank = " << getParent()->icCommunicator()->commRank() << std::endl;
#endif
#endif
      status = this->normalizeFamily();
      assert( (status == PV_SUCCESS) || (status == PV_BREAK));
   }

   return PV_BREAK;
} // normalizeWeights

} // namespace PV
