/*
 * SiblingConn.cpp
 *
 *  Created on: Jan 26, 2012
 *      Author: garkenyon
 */

#include "SiblingConn.hpp"

namespace PV {

SiblingConn::SiblingConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
      const char * filename, InitWeights *weightInit, SiblingConn *sibling_conn)
{
   SiblingConn::initialize_base();
   SiblingConn::initialize(name, hc, pre, post, filename, weightInit, sibling_conn);
   // HyPerConn::initialize is not virtual
}

int SiblingConn::initialize(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, const char * filename,
      InitWeights *weightInit, SiblingConn *sibling_conn)
{
   siblingConn = sibling_conn;
   isNormalized = false;
   if (siblingConn != NULL){
      siblingConn->setSiblingConn(this);
   }
   return KernelConn::initialize(name, hc, pre, post, filename, weightInit);
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

int SiblingConn::normalizeFamily() {
	// normalize all arbors individuqlly relative to siblings
	// insert MPI_Barrier before callng SiblingConn::normalizeFamily to ensure all processors have finished self normalization
	const int num_kernels = getNumDataPatches();
	double sum_local = 0.0;
	double sum_sibling = 0.0;
	for (int kArbor = 0; kArbor < this->numberOfAxonalArborLists(); kArbor++) {
#ifdef USE_SHMGET
		if (shmget_flag && (!shmget_owner[kArbor] || !siblingConn->getShmgetOwner(kArbor))) {
			continue;
		}
#endif
		for (int kPatch = 0; kPatch < num_kernels; kPatch++) {
			// PVPatch * localWPatch = getWeights(kPatch,kArbor); // this->getKernelPatch(kArbor, kPatch);
			// pvdata_t * myWeights = myWpatch->data;
			pvdata_t * localWeights = get_wDataHead(kArbor, kPatch);
			assert(localWeights != NULL);
			// PVPatch * siblingWpatch = siblingConn->getKernelPatch(kArbor, kPatch);
			// pvdata_t * siblingWeights = siblingWpatch->data;
#ifdef USE_SHMGET
			volatile pvdata_t * siblingWeights = siblingConn->get_wDataHead(
					kArbor, kPatch);
#else
			pvdata_t * siblingWeights = siblingConn->get_wDataHead(kArbor, kPatch);
#endif
			assert(siblingWeights != NULL);
			const int nx = nxp; // localWPatch->nx;
			const int ny = nyp; // localWPatch->ny;
			const int nf = nfp;
			const int sy = syp;
			for (int ky = 0; ky < ny; ky++) {
				for (int iWeight = 0; iWeight < nf * nx; iWeight++) {
					pvdata_t norm_denom = fabs(siblingWeights[iWeight])
							+ fabs(localWeights[iWeight]);
					norm_denom = (norm_denom != 0.0f) ? norm_denom : 1.0f;
#ifdef USE_SHMGET
					if (shmget_flag){
						if (shmget_owner[kArbor]) {
							localWeights[iWeight] /= norm_denom;
						}
					}
					else{
						localWeights[iWeight] /= norm_denom;
					}
#else
					localWeights[iWeight] /= norm_denom;
#endif
#ifdef USE_SHMGET
					if (shmget_flag){
						if (siblingConn->getShmgetOwner(kArbor)) {
							siblingWeights[iWeight] /= norm_denom;
						}
					}
					else{
						siblingWeights[iWeight] /= norm_denom;
					}
#else
					siblingWeights[iWeight] /= norm_denom;
#endif
#ifdef PV_USE_MPI
#ifdef USE_SHMGET
		// insert barrier to ensure that all processes have finished updating weights before computing sum
					MPI_Barrier(parent->icCommunicator()->communicator());
#endif
#endif
					sum_local += localWeights[iWeight];
					sum_sibling += siblingWeights[iWeight];
				}
				localWeights += sy;
				siblingWeights += sy;
			}
		} // kPatch < numPatches
	} // kArbor
	  // scale local weights so that average = sibling average
	float scale_factor =
			sum_local != 0 ? fabs(sum_sibling) / fabs(sum_local) : 1;
	for (int kArbor = 0; kArbor < this->numberOfAxonalArborLists(); kArbor++) {
#ifdef USE_SHMGET
		if (shmget_flag && !shmget_owner[kArbor]) {
			continue;
		}
		volatile pvdata_t * localWeights = this->get_wDataStart(kArbor);
#else
		pvdata_t * localWeights = this->get_wDataStart(kArbor);
#endif
		for (int iWeight = 0; iWeight < nxp * nyp * nfp; iWeight++) {
			localWeights[iWeight] *= scale_factor;
		}
	}
	return PV_BREAK;
} // normalizeFamily

int SiblingConn::normalizeWeights(PVPatch ** patches, pvdata_t ** dataStart,
		int numPatches, int arborId) {
	int status = PV_SUCCESS;

	status = NoSelfKernelConn::normalizeWeights(patches, dataStart, numPatches,
			arborId);  // parent class should return PV_BREAK
	assert( (status == PV_SUCCESS) || (status == PV_BREAK));

	if ((siblingConn != NULL) && (siblingConn->getIsNormalized())) {
#ifdef PV_USE_MPI
#ifdef USE_SHMGET
		// insert barrier to ensure that all processes have finished individual normalization before starting family normalization
		MPI_Barrier(parent->icCommunicator()->communicator());
#endif
#endif
		status = this->normalizeFamily();
		assert( (status == PV_SUCCESS) || (status == PV_BREAK));
	}

	return PV_BREAK;
} // normalizeWeights


} // namespace PV
