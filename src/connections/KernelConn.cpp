/*
 * KernelConn.cpp
 *
 *  Created on: Aug 6, 2009
 *      Author: gkenyon
 */

#include "KernelConn.hpp"
#include <assert.h>
#include <float.h>
#include "../io/io.h"
#ifdef USE_SHMGET
   #include <sys/shm.h>
#endif

namespace PV {

KernelConn::KernelConn()
{
   initialize_base();
}


KernelConn::KernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
      const char * filename, InitWeights *weightInit) : HyPerConn()
{
   KernelConn::initialize_base();
   KernelConn::initialize(name, hc, pre, post, filename, weightInit);
   // HyPerConn::initialize is not virtual
#ifdef PV_USE_OPENCL
   if(gpuAccelerateFlag)
      initializeGPU();
#endif
}

KernelConn::~KernelConn() {
   deleteWeights();
   #ifdef PV_USE_MPI
   free(mpiReductionBuffer);
#endif // PV_USE_MPI
}

int KernelConn::initialize_base()
{
   fileType = PVP_KERNEL_FILE_TYPE;
   lastUpdateTime = 0.f;
   plasticityFlag = false;
   normalizeArborsIndividually = false;
   symmetrizeWeightsFlag = false;
   // nxKernel = 0;
   // nyKernel = 0;
   // nfKernel = 0;
#ifdef PV_USE_MPI
   keepKernelsSynchronized_flag = true;
   mpiReductionBuffer = NULL;
#endif // PV_USE_MPI
   return PV_SUCCESS;
   // KernelConn constructor calls HyPerConn::HyPerConn(), which
   // calls HyPerConn::initialize_base().
}

int KernelConn::initialize(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, const char * filename,
      InitWeights *weightInit)
{
   PVParams * params = hc->parameters();
   bool normalize_flag = params->value(name, "normalize",0,true);
   if (normalize_flag){
      symmetrizeWeightsFlag = params->value(name, "symmetrizeWeights",0);
   }
#ifdef USE_SHMGET
	bool plasticity_flag = params->value(name, "plasticityFlag", plasticityFlag,
			true) != 0;
	shmget_flag = params->value(name, "shmget_flag", shmget_flag, true);
//#ifdef PV_USE_MPI
//	if (parent->icCommunicator() != NULL
//			&& parent->icCommunicator()->numCommColumns()
//					* parent->icCommunicator()->numCommRows() > 1
//			&& shmget_flag) {
//		std::cout << "warning: in KernelConn::initialize: " << this->name
//				<< ", shmget_flag parameter specified as true but only 1 process"
//				<< std::endl;
//	}
//#endif
	if (plasticity_flag && shmget_flag) {
		shmget_flag = false;
		std::cout << "in KernelConn::initialize: " << this->name
				<< ", shmget_flag parameter specified as true, reset to false because plasticity_flag is true"
				<< std::endl;
	}
#endif
   HyPerConn::initialize(name, hc, pre, post, filename, weightInit);
   initializeUpdateTime(params); // sets weightUpdatePeriod and initial value of weightUpdateTime
   lastUpdateTime = weightUpdateTime - parent->getDeltaTime();

   // nxKernel = (pre->getXScale() < post->getXScale()) ? pow(2,
   //       post->getXScale() - pre->getXScale()) : 1;
   // nyKernel = (pre->getYScale() < post->getYScale()) ? pow(2,
   //       post->getYScale() - pre->getYScale()) : 1;
   // nfKernel = pre->getLayerLoc()->nf;

#ifdef PV_USE_MPI
   // preallocate buffer for MPI_Allreduce call in reduceKernels
   //int arborID = 0; // for now, only one axonal arbor
   const int numPatches = getNumDataPatches();
   const size_t patchSize = nxp*nyp*nfp*sizeof(pvdata_t);
   const size_t localSize = numPatches * patchSize;
   mpiReductionBuffer = (pvdata_t *) malloc(localSize*sizeof(pvdata_t));
   if(mpiReductionBuffer == NULL) {
      fprintf(stderr, "KernelConn::initialize:Unable to allocate memory\n");
      exit(PV_FAILURE);
   }
#endif // PV_USE_MPI
#ifdef PV_USE_OPENCL
//   //don't support GPU accelleration in kernelconn yet
//   ignoreGPUflag=false;
//   //tell the recieving layer to copy gsyn to the gpu, because kernelconn won't be calculating it
//   post->copyChannelToDevice();
#endif

   initPatchToDataLUT();

   return PV_SUCCESS;
}

int KernelConn::createArbors() {
#ifdef USE_SHMGET
	if (shmget_flag){
		shmget_id = (int *) calloc(this->numberOfAxonalArborLists(),
				sizeof(int));
		assert(shmget_id != NULL);
		shmget_owner = (bool *) calloc(this->numberOfAxonalArborLists(),
				sizeof(bool));
		assert(shmget_owner != NULL);
	}
#endif
   HyPerConn::createArbors();
   return PV_SUCCESS; //should we check if allocation was successful?
}

int KernelConn::initPlasticityPatches() {
   if( getPlasticityFlag() ) {
#ifdef PV_USE_MPI
   keepKernelsSynchronized_flag = getParent()->parameters()->value(name, "keepKernelsSynchronized", keepKernelsSynchronized_flag, true);
#endif
      HyPerConn::initPlasticityPatches();
   }
   return PV_SUCCESS;
}

int KernelConn::initializeUpdateTime(PVParams * params) {
   if( plasticityFlag ) {
      float defaultUpdatePeriod = 1.f;
      weightUpdatePeriod = params->value(name, "weightUpdatePeriod", defaultUpdatePeriod);
      weightUpdateTime = params->value(name, "initialWeightUpdateTime", 0.0f);
   }
   else {
      weightUpdateTime = 0.0f;
   }
   return PV_SUCCESS;
}

// use shmget() to save memory on shared memory architectures
pvdata_t * KernelConn::allocWeights(PVPatch *** patches, int nPatches,
		int nxPatch, int nyPatch, int nfPatch, int arbor_ID) {
	int sx = nfPatch;
	int sy = sx * nxPatch;
	int sp = sy * nyPatch;

	size_t patchSize = sp * sizeof(pvdata_t);
	size_t dataSize = nPatches * patchSize;

	pvdata_t * dataPatches = NULL; // (pvdata_t *) calloc(dataSize, sizeof(char));
#ifdef USE_SHMGET

	if (!shmget_flag) {
		dataPatches = (pvdata_t *) calloc(dataSize, sizeof(char));
//      shmget_flag = false;
//      shmget_owner[arbor_ID] = true;
	} else {
		shmget_owner[arbor_ID] = true;
		// shmget diagnostics
#define SHMGET_DEBUG
#ifdef SHMGET_DEBUG
		int rank_tmp = parent->icCommunicator()->commRank();
		if (arbor_ID == 0 || arbor_ID == this->numberOfAxonalArborLists()) {
			std::cout << "rank = " << rank_tmp;
			std::cout << ", arbor_ID = " << arbor_ID;
		}
#endif // SHMGET_DEBUG
		// dataSize must be a multiple of PAGE_SIZE
		size_t shmget_dataSize = (floor(dataSize / PAGE_SIZE) + 1) * PAGE_SIZE;
		key_t key = IPC_PRIVATE;
		const int max_arbors = 8712;
		key = 11 + (this->getConnectionId() + 1) * max_arbors + arbor_ID; //hopefully unique key identifier for all shared memory associated with this connection arbor
		int shmflg = (IPC_CREAT | IPC_EXCL | 0666);
		char *segptr;

		// check for existing segment associated with this key, delete existing segment if present, then insert barrier to ensure
		// all processes have completed this check before attempting to create new shared memory segment
		int shmget_existing_ID = shmget(key, shmget_dataSize, 0666);
		if (shmget_existing_ID != -1){
			shmid_ds * shmget_ds = NULL;
			int shmget_status = shmctl(shmget_existing_ID, IPC_RMID,
					shmget_ds);
			assert(shmget_status==0);
		}
#ifdef PV_USE_MPI
         MPI_Barrier(getParent()->icCommunicator()->communicator());
#endif // PV_USE_MPI


 		/* Open the shared memory segment - create if necessary */
		if ((shmget_id[arbor_ID] = shmget(key, shmget_dataSize, shmflg))
				== -1) {
			if (errno != EEXIST) {
				std::cout << std::endl;
				std::cout << "key = " << key << ", shmget_dataSize = "
						<< shmget_dataSize << ", shmflg = "
						<< shmflg << std::endl;
				perror("shmget: unable to create shared memory segment");
				exit(1);
			}
			/* Segment already exists - try as a client */
			shmget_owner[arbor_ID] = false;
			int shmget_flag2 = (IPC_CREAT | 0666);
			if ((shmget_id[arbor_ID] = shmget(key, shmget_dataSize,
					shmget_flag2)) == -1) {
				perror(
						"shmget: unable to obtain id of existing shared memory segment");
				exit(1);
			}
		}
#ifdef SHMGET_DEBUG
		if (arbor_ID % 100 == 0) {
			std::cout << ", shmget_owner = " << shmget_owner[arbor_ID]
					<< std::endl;
		}
#endif // SHMGET_DEBUG
		/* Attach (map) the shared memory segment into the current process */
		if ((segptr = (char *) shmat(shmget_id[arbor_ID], 0, 0))
				== (char *) -1) {
			perror("shmat: unable to map shared memory segment");
			exit(1);
		}
		dataPatches = (pvdata_t *) segptr;
	}
#else
	dataPatches = (pvdata_t *) calloc(dataSize, sizeof(char));
#endif // USE_SHMGET
	assert(dataPatches != NULL);

	return dataPatches;
}



int KernelConn::deleteWeights()
{
   free(patch2datalookuptable);
   return 0; // HyPerConn destructor will call HyPerConn::deleteWeights()

}

PVPatch ***  KernelConn::initializeWeights(PVPatch *** patches, pvdata_t ** dataStart, int numPatches,
      const char * filename)
{
   //int arbor = 0;
   int numKernelPatches = getNumDataPatches();
   HyPerConn::initializeWeights(NULL, dataStart, numKernelPatches, filename);
   return patches;
}

//PVPatch ** KernelConn::readWeights(PVPatch ** patches, int numPatches,
//      const char * filename)
//{
//   //HyPerConn::readWeights(patches, numPatches, filename);
//
//   return patches;
//}

int KernelConn::initNumDataPatches()
{
   int nxKernel = (pre->getXScale() < post->getXScale()) ? (int)pow(2,
         post->getXScale() - pre->getXScale()) : 1;
   int nyKernel = (pre->getYScale() < post->getYScale()) ? (int)pow(2,
         post->getYScale() - pre->getYScale()) : 1;
   numDataPatches = pre->clayer->loc.nf * nxKernel * nyKernel;
   return PV_SUCCESS;
}

float KernelConn::minWeight(int arborId)
{
   const int numKernels = getNumDataPatches();
   const int numWeights = nxp * nyp * nfp;
   float min_weight = FLT_MAX;
   for (int iKernel = 0; iKernel < numKernels; iKernel++) {
      pvdata_t * kernelWeights = this->get_wDataHead(arborId, iKernel);
      for (int iWeight = 0; iWeight < numWeights; iWeight++) {
         min_weight = (min_weight < kernelWeights[iWeight]) ? min_weight
               : kernelWeights[iWeight];
      }
   }
   return min_weight;
}

float KernelConn::maxWeight(int arborId)
{
   const int numKernels = getNumDataPatches();
   const int numWeights = nxp * nyp * nfp;
   float max_weight = -FLT_MAX;
   for (int iKernel = 0; iKernel < numKernels; iKernel++) {
      pvdata_t * kernelWeights = this->get_wDataHead(arborId, iKernel);
      for (int iWeight = 0; iWeight < numWeights; iWeight++) {
         max_weight = (max_weight > kernelWeights[iWeight]) ? max_weight
               : kernelWeights[iWeight];
      }
   }
   return max_weight;
}

int KernelConn::calc_dW(int arbor_ID){
   clear_dW(arbor_ID);
   update_dW(arbor_ID);
   return PV_BREAK;
}

int KernelConn::clear_dW(int arbor_ID) {
   // zero dwDataStart
   for(int kArbor = 0; kArbor < numberOfAxonalArborLists(); kArbor++){
      for(int kKernel = 0; kKernel < getNumDataPatches(); kKernel++){
         int syPatch = syp;
         int nkPatch = nfp * nxp;
         float * dWeights = get_dwDataHead(arbor_ID,kKernel);
         for(int kyPatch = 0; kyPatch < nyp; kyPatch++){
            for(int kPatch = 0; kPatch < nkPatch; kPatch++){
               dWeights[kPatch] = 0.0f;
            }
            dWeights += syPatch;
         }
      }
   }
   return PV_BREAK;
   //return PV_SUCCESS;
}
int KernelConn::update_dW(int arbor_ID) {
   // Typically override this method with a call to defaultUpdate_dW(arbor_ID)
   return PV_BREAK;
   //return PV_SUCCESS;
}

int KernelConn::defaultUpdate_dW(int arbor_ID) {
   // compute dW but don't add them to the weights yet.
   // That takes place in reduceKernels, so that the output is
   // independent of the number of processors.
   int nExt = preSynapticLayer()->getNumExtended();
   int numKernelIndices = getNumDataPatches();
   const pvdata_t * preactbuf = preSynapticLayer()->getLayerData(getDelay(arbor_ID));
   const pvdata_t * postactbuf = postSynapticLayer()->getLayerData(getDelay(arbor_ID));

   int sya = (post->getLayerLoc()->nf * (post->getLayerLoc()->nx + 2*post->getLayerLoc()->nb));

   for(int kExt=0; kExt<nExt;kExt++) {
      PVPatch * weights = getWeights(kExt,arbor_ID);
      size_t offset = getAPostOffset(kExt, arbor_ID);
      pvdata_t preact = preactbuf[kExt];
      int ny = weights->ny;
      int nk = weights->nx * nfp;
      const pvdata_t * postactRef = &postactbuf[offset];
      pvdata_t * dwdata = get_dwData(arbor_ID, kExt);
      int lineoffsetw = 0;
      int lineoffseta = 0;
      for( int y=0; y<ny; y++ ) {
         for( int k=0; k<nk; k++ ) {
            dwdata[lineoffsetw + k] += updateRule_dW(preact, postactRef[lineoffseta+k]);
         }
         lineoffsetw += syp;
         lineoffseta += sya;
      }
   }

   // Divide by (numNeurons/numKernels)
   int divisor = pre->getNumNeurons()/numKernelIndices;
   assert( divisor*numKernelIndices == pre->getNumNeurons() );
   for( int kernelindex=0; kernelindex<numKernelIndices; kernelindex++ ) {
      int numpatchitems = nxp*nyp*nfp;
      pvdata_t * dwpatchdata = get_dwDataHead(arbor_ID,kernelindex);
      for( int n=0; n<numpatchitems; n++ ) {
         dwpatchdata[n] /= divisor;
      }
   }

   lastUpdateTime = parent->simulationTime();

   return PV_SUCCESS;
}

pvdata_t KernelConn::updateRule_dW(pvdata_t pre, pvdata_t post) {
   return dWMax * pre*post;
}

int KernelConn::updateState(double timef, double dt) {
   int status = PV_SUCCESS;
   if( !plasticityFlag ) {
      return status;
   }
   update_timer->start();
   if( timef >= weightUpdateTime) {
      computeNewWeightUpdateTime(timef, weightUpdateTime);
      for(int arborID=0;arborID<numberOfAxonalArborLists();arborID++) {
         status = calc_dW(arborID);  // calculate changes in weights
         if (status == PV_BREAK) {break;}
         assert(status == PV_SUCCESS);
      }

#ifdef PV_USE_MPI
      if (keepKernelsSynchronized_flag
            || parent->simulationTime() >= parent->getStopTime()-parent->getDeltaTime()) {
         for (int arborID = 0; arborID < numberOfAxonalArborLists(); arborID++) {
            status = reduceKernels(arborID); // combine partial changes in each column
            if (status == PV_BREAK) {
               break;
            }
            assert(status == PV_SUCCESS);
         }
      }
#endif // PV_USE_MPI

      for(int arborID=0;arborID<numberOfAxonalArborLists();arborID++) {
         status = updateWeights(arborID);  // calculate new weights from changes
         if (status == PV_BREAK) {break;}
         assert(status == PV_SUCCESS);
      }
      if( normalize_flag ) {
         for(int arborID=0;arborID<numberOfAxonalArborLists();arborID++) {
            status = normalizeWeights(NULL, this->get_wDataStart(), getNumDataPatches(), arborID);
            if (status == PV_BREAK) {break;}
            assert(status == PV_SUCCESS);
         }
      }  // normalize_flag
   } // time > weightUpdateTime

update_timer->stop();
return PV_SUCCESS;
} // updateState

int KernelConn::updateWeights(int arbor_ID){
   lastUpdateTime = parent->simulationTime();
   // add dw to w
#ifdef USE_SHMGET
   if (shmget_flag && !shmget_owner[arbor_ID]){
         return PV_BREAK;
      }
#endif
   for(int kArbor = 0; kArbor < this->numberOfAxonalArborLists(); kArbor++){
#ifdef USE_SHMGET
      volatile pvdata_t * w_data_start = get_wDataStart(kArbor);
#else
      pvdata_t * w_data_start = get_wDataStart(kArbor);
#endif
      for( int k=0; k<nxp*nyp*nfp*getNumDataPatches(); k++ ) {
         w_data_start[k] += get_dwDataStart(kArbor)[k];
      }
   }
   return PV_BREAK;
}

float KernelConn::computeNewWeightUpdateTime(double time, double currentUpdateTime) {
   // Is only called by KernelConn::updateState if plasticityFlag is true
   weightUpdateTime += weightUpdatePeriod;
   return weightUpdateTime;
}

#ifdef PV_USE_MPI
int KernelConn::reduceKernels(const int arborID) {
   Communicator * comm = parent->icCommunicator();
   const MPI_Comm mpi_comm = comm->communicator();
   int ierr;
   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();
   const int nProcs = nxProcs * nyProcs;
   if (nProcs == 1){
      return PV_BREAK;
   }
   const int numPatches = getNumDataPatches();
   const size_t patchSize = nxp*nyp*nfp*sizeof(pvdata_t);
   const size_t localSize = numPatches * patchSize;

   // Copy this column's weights into mpiReductionBuffer
   int idx = 0;
   for (int k = 0; k < numPatches; k++) {
      const pvdata_t * data = get_dwDataHead(arborID,k);

      for (int y = 0; y < nyp; y++) {
         for (int x = 0; x < nxp; x++) {
            for (int f = 0; f < nfp; f++) {
               mpiReductionBuffer[idx] = data[x*sxp + y*syp + f*sfp];
               idx++;
            }
         }
      }
   }

   // MPI_Allreduce combines all processors' buffers and puts the common result
   // into each processor's buffer.
   ierr = MPI_Allreduce(MPI_IN_PLACE, mpiReductionBuffer, localSize, MPI_FLOAT, MPI_SUM, mpi_comm);
   // TODO error handling

   // mpiReductionBuffer now holds the sum over all processes.
   // Divide by number of processes to get average and copy back to patches
   idx = 0;
   for (int k = 0; k < numPatches; k++) {
      pvdata_t * data = get_dwDataHead(arborID,k); // p->data;

      for (int y = 0; y < nyp; y++) {
         for (int x = 0; x < nxp; x++) {
            for (int f = 0; f < nfp; f++) {
               data[x*sxp + y*syp + f*sfp] = mpiReductionBuffer[idx]/nProcs;
               idx++;
            }
         }
      }
   }

   return PV_SUCCESS;
}
#endif // PV_USE_MPI

void KernelConn::initPatchToDataLUT() {
   int numWeightPatches=getNumWeightPatches();

   patch2datalookuptable=(int *) calloc(numWeightPatches, sizeof(int));
   for(int i=0; i<numWeightPatches; i++) {
      int kernelindex=patchIndexToDataIndex(i);
      patch2datalookuptable[i]=kernelindex;
   }

}
int KernelConn::patchToDataLUT(int patchIndex) {
   // This uses a look-up table
   return patch2datalookuptable[patchIndex];
}

int KernelConn::normalizeWeights(PVPatch ** patches, pvdata_t ** dataStart,
		int numPatches, int arborId) {
	int status = PV_SUCCESS;
	const int num_kernels = getNumDataPatches();

	// symmetrize before normalization
	if (symmetrizeWeightsFlag) {
		for (int kArbor = 0; kArbor < this->numberOfAxonalArborLists();
				kArbor++) {
			status = symmetrizeWeights(dataStart[arborId], num_kernels, kArbor);
			assert( (status == PV_SUCCESS) || (status == PV_BREAK));
		}
	}
	status = HyPerConn::normalizeWeights(NULL, dataStart, num_kernels, arborId);
	assert( (status == PV_SUCCESS) || (status == PV_BREAK));
	return status;
}

int KernelConn::symmetrizeWeights(pvdata_t * dataStart, int numPatches, int arborId)
{
   int status = PV_SUCCESS;
   if (arborId == 0)
      printf("Entering KernelConn::symmetrizeWeights for connection \"%s\"\n", name);
   assert(pre->clayer->loc.nf==post->clayer->loc.nf);

   pvdata_t * symPatches = (pvdata_t *) calloc(nxp*nyp*nfp*numPatches, sizeof(pvdata_t));
   assert(symPatches != NULL);

   const int sy = nxp * nfp;
   const float deltaTheta = PI / nfp;
   const float offsetTheta = 0.5f * deltaTheta;
   const int kyMid = nyp / 2;
   const int kxMid = nxp / 2;
   for (int iSymKernel = 0; iSymKernel < numPatches; iSymKernel++) {
      pvdata_t * symW = symPatches + iSymKernel*nxp*nyp*nfp;
      float symTheta = offsetTheta + iSymKernel * deltaTheta;
      for (int kySym = 0; kySym < nyp; kySym++) {
         float dySym = kySym - kyMid;
         for (int kxSym = 0; kxSym < nxp; kxSym++) {
            float dxSym = kxSym - kxMid;
            float distSym = sqrt(dxSym * dxSym + dySym * dySym);
            if (distSym > abs(kxMid > kyMid ? kxMid : kyMid)) {
               continue;
            }
            float dyPrime = dySym * cos(symTheta) - dxSym * sin(symTheta);
            float dxPrime = dxSym * cos(symTheta) + dySym * sin(symTheta);
            for (int kfSym = 0; kfSym < nfp; kfSym++) {
               int kDf = kfSym - iSymKernel;
               int iSymW = kfSym + nfp * kxSym + sy * kySym;
               for (int iKernel = 0; iKernel < nfp; iKernel++) {
                  // PVPatch * kerWp = getKernelPatch(arborId, iKernel);
                  pvdata_t * kerW = get_wDataStart(arborId) + iKernel*nxp*nyp*nfp;
                  int kfRot = iKernel + kDf;
                  if (kfRot < 0) {
                     kfRot = nfp + kfRot;
                  }
                  else {
                     kfRot = kfRot % nfp;
                  }
                  float rotTheta = offsetTheta + iKernel * deltaTheta;
                  float yRot = dyPrime * cos(rotTheta) + dxPrime * sin(rotTheta);
                  float xRot = dxPrime * cos(rotTheta) - dyPrime * sin(rotTheta);
                  yRot += kyMid;
                  xRot += kxMid;
                  // should find nearest neighbors and do weighted average
                  int kyRot = yRot + 0.5f;
                  int kxRot = xRot + 0.5f;
                  int iRotW = kfRot + nfp * kxRot + sy * kyRot;
                  symW[iSymW] += kerW[iRotW] / nfp;
               } // kfRot
            } // kfSymm
         } // kxSym
      } // kySym
   } // iKernel
   const int num_weights = nfp * nxp * nyp;
   for (int iKernel = 0; iKernel < numPatches; iKernel++) {
      pvdata_t * kerW = get_wDataStart(arborId)+iKernel*nxp*nyp*nfp;
      pvdata_t * symW = symPatches + iKernel*nxp*nyp*nfp;
      for (int iW = 0; iW < num_weights; iW++) {
         kerW[iW] = symW[iW];
      }
   } // iKernel

   free(symPatches);

   printf("Exiting KernelConn::symmetrizeWeights for connection \"%s\"\n", name);
   return status;
}

int KernelConn::writeWeights(double timef, bool last) {
   const int numPatches = getNumDataPatches();
   return HyPerConn::writeWeights(NULL, get_wDataStart(), numPatches, NULL, timef, writeCompressedWeights, last);
}

int KernelConn::writeWeights(const char * filename) {
   return HyPerConn::writeWeights(NULL, get_wDataStart(), getNumDataPatches(), filename, parent->simulationTime(), writeCompressedWeights, /*last*/true);
}

int KernelConn::checkpointRead(const char * cpDir, double * timef) {
   // Only difference from HyPerConn::checkpointRead() is first argument to weightsInitObject->initializeWeights.
   // Can we juggle things so that KernelConn::checkpointWrite is unnecessary?
   clearWeights(get_wDataStart(), getNumDataPatches(), nxp, nyp, nfp);

   char path[PV_PATH_MAX];
   int status = checkpointFilename(path, PV_PATH_MAX, cpDir);
   assert(status==PV_SUCCESS);
   InitWeights * weightsInitObject = new InitWeights();
   weightsInitObject->initializeWeights(NULL, get_wDataStart(), getNumDataPatches(), path, this, timef);
   return PV_SUCCESS;
}

int KernelConn::checkpointWrite(const char * cpDir) {
   char filename[PV_PATH_MAX];
   int status = checkpointFilename(filename, PV_PATH_MAX, cpDir);
   assert(status==PV_SUCCESS);
#ifdef PV_USE_MPI
   if (!keepKernelsSynchronized_flag) {
      for (int arbor_id = 0; arbor_id < this->numberOfAxonalArborLists(); arbor_id++) {
         reduceKernels(arbor_id);
      }
   }
#endif // PV_USE_MPI
   return HyPerConn::writeWeights(NULL, get_wDataStart(), getNumDataPatches(), filename, parent->simulationTime(), writeCompressedCheckpoints, /*last*/true);
}

int KernelConn::patchIndexToDataIndex(int patchIndex, int * kx/*default=NULL*/, int * ky/*default=NULL*/, int * kf/*default=NULL*/) {
   return calcUnitCellIndex(patchIndex, kx, ky, kf);
}

int KernelConn::dataIndexToUnitCellIndex(int dataIndex, int * kx/*default=NULL*/, int * ky/*default=NULL*/, int * kf/*default=NULL*/) {
   int nfUnitCell = pre->getLayerLoc()->nf;
   int nxUnitCell = zUnitCellSize(pre->getXScale(), post->getXScale());
   int nyUnitCell = zUnitCellSize(pre->getYScale(), post->getYScale());
   assert( dataIndex >= 0 && dataIndex < nxUnitCell*nyUnitCell*nfUnitCell );
   if(kx) *kx = kxPos(dataIndex, nxUnitCell, nyUnitCell, nfUnitCell);
   if(ky) *ky = kyPos(dataIndex, nxUnitCell, nyUnitCell, nfUnitCell);
   if(kf) *kf = featureIndex(dataIndex, nxUnitCell, nyUnitCell, nfUnitCell);
   return dataIndex;
}

int KernelConn::getReciprocalWgtCoordinates(int kx, int ky, int kf, int kernelidx, int * kxRecip, int * kyRecip, int * kfRecip, int * kernelidxRecip) {
   int status = PV_SUCCESS;

   assert( kx>=0 && kx<nxp && ky>=0 && ky<nyp && kf>=0 && kf<nfp && kernelidx>=0 && kernelidx<getNumDataPatches());
   if( status == PV_SUCCESS ) {
      int nxUnitCell = zUnitCellSize(pre->getXScale(), post->getXScale());
      int nyUnitCell = zUnitCellSize(pre->getYScale(), post->getYScale());
      int nfUnitCell = pre->getLayerLoc()->nf;

      int nxUnitCellRecip = zUnitCellSize(post->getXScale(), pre->getXScale());
      int nyUnitCellRecip = zUnitCellSize(post->getYScale(), pre->getYScale());
      int nfUnitCellRecip = post->getLayerLoc()->nf;

      double xScaleFactor = pow(2,pre->getXScale()-post->getXScale()); // many-to-one connections have xScaleFactor<1; one-to-many, >1.
      double yScaleFactor = pow(2,pre->getYScale()-post->getYScale());

      int kxKern = kxPos(kernelidx, nxUnitCell, nyUnitCell, nfUnitCell);
      int kyKern = kyPos(kernelidx, nxUnitCell, nyUnitCell, nfUnitCell);
      int kfKern = featureIndex(kernelidx, nxUnitCell, nyUnitCell, nfUnitCell);

      int xInPostCell = kx % nxUnitCellRecip;
      int yInPostCell = ky % nyUnitCellRecip;

      *kxRecip = (int) ((nxp-1-kx)/xScaleFactor) + kxKern;
      *kyRecip = (int) ((nyp-1-ky)/yScaleFactor) + kyKern;
      *kfRecip = kfKern;
      *kernelidxRecip = kIndex(xInPostCell, yInPostCell, kf, nxUnitCellRecip, nyUnitCellRecip, nfUnitCellRecip);
   }

   return status;
}


} // namespace PV

