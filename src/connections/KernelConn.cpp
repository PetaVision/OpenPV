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

namespace PV {

KernelConn::KernelConn()
{
   initialize_base();
}


KernelConn::KernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
      ChannelType channel, const char * filename, InitWeights *weightInit) : HyPerConn()
{
   KernelConn::initialize_base();
   KernelConn::initialize(name, hc, pre, post, channel, filename, weightInit);
   // HyPerConn::initialize is not virtual
}

KernelConn::~KernelConn() {
   free(kernelPatches);
   if (dKernelPatches != NULL) {free(dKernelPatches);};
#ifdef PV_USE_MPI
   free(mpiReductionBuffer);
#endif // PV_USE_MPI
}

int KernelConn::initialize_base()
{
   kernelPatches = NULL;
   dKernelPatches = NULL;
   fileType = PVP_KERNEL_FILE_TYPE;
   lastUpdateTime = 0.f;
   plasticityFlag = false;
   tmpPatch = NULL;
   this->normalize_arbors_individually = false;
   nxKernel = 0;
   nyKernel = 0;
   nfKernel = 0;
#ifdef PV_USE_MPI
   mpiReductionBuffer = NULL;
#endif // PV_USE_MPI
   return PV_SUCCESS;
   // KernelConn constructor calls HyPerConn::HyPerConn(), which
   // calls HyPerConn::initialize_base().
}

int KernelConn::initialize(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, ChannelType channel, const char * filename,
      InitWeights *weightInit)
{
   PVParams * params = hc->parameters();
   symmetrizeWeightsFlag = params->value(name, "symmetrizeWeights",0);
   HyPerConn::initialize(name, hc, pre, post, channel, filename, weightInit);
   weightUpdateTime = initializeUpdateTime(params);
   lastUpdateTime = weightUpdateTime - parent->getDeltaTime();

   nxKernel = (pre->getXScale() < post->getXScale()) ? pow(2,
         post->getXScale() - pre->getXScale()) : 1;
   nyKernel = (pre->getYScale() < post->getYScale()) ? pow(2,
         post->getYScale() - pre->getYScale()) : 1;
   nfKernel = pre->getLayerLoc()->nf;

#ifdef PV_USE_MPI
   // preallocate buffer for MPI_Allreduce call in reduceKernels
   //int axonID = 0; // for now, only one axonal arbor
   const int numPatches = numDataPatches();
   const size_t patchSize = nxp*nyp*nfp*sizeof(pvdata_t);
   const size_t localSize = numPatches * patchSize;
   mpiReductionBuffer = (pvdata_t *) malloc(localSize*sizeof(pvdata_t));
   if(mpiReductionBuffer == NULL) {
      fprintf(stderr, "KernelConn::initialize:Unable to allocate memory\n");
      exit(PV_FAILURE);
   }
#endif // PV_USE_MPI
   return PV_SUCCESS;
}

int KernelConn::createArbors() {
   HyPerConn::createArbors();
   kernelPatches = (PVPatch***) calloc(numberOfAxonalArborLists(), sizeof(PVPatch**));
   for (int arborId = 0; arborId < numberOfAxonalArborLists(); arborId++) {
      kernelPatches[arborId] = NULL;
   }
   assert(kernelPatches!=NULL);
   if (this->getPlasticityFlag()){
      dKernelPatches = (PVPatch***) calloc(numberOfAxonalArborLists(), sizeof(PVPatch**));
      assert(dKernelPatches!=NULL);
      for (int arborId = 0; arborId < numberOfAxonalArborLists(); arborId++) {
         dKernelPatches[arborId] = NULL;
      }
   }
   return PV_SUCCESS; //should we check if allocation was successful?
}

int KernelConn::setWPatches(PVPatch ** patches, int arborId){
   int status = HyPerConn::setWPatches(patches, arborId);
   assert(status == 0);
   assert(kernelPatches[arborId] == NULL);
   kernelPatches[arborId] = tmpPatch;  // tmpPatch stores most recently allocated PVPatch**
   tmpPatch = NULL;  // assert(tmpPatch == NULL) before assigning tmpPatch;
   return PV_SUCCESS;
}

int KernelConn::setdWPatches(PVPatch ** patches, int arborId){
   int status = HyPerConn::setdWPatches(patches, arborId);
   assert(status == 0);
   assert(dKernelPatches[arborId] == NULL);
   dKernelPatches[arborId] = tmpPatch;  // tmpPatch stores most recently allocated PVPatch**
   tmpPatch = NULL;  // assert(tmpPatch == NULL) before assigning tmpPatch;
   return PV_SUCCESS;
}

pvdata_t * KernelConn::allocWeights(PVPatch *** patches, int nPatches, int nxPatch,
      int nyPatch, int nfPatch, int axonId)
{
   //const int arbor = 0;
   int numKernelPatches = numDataPatches();

// allocate kernel (or dKernelPatches)
   assert(tmpPatch == NULL);
//   tmpPatch = (PVPatch**) calloc(sizeof(PVPatch*), numKernelPatches);
   tmpPatch = (PVPatch**) calloc(numKernelPatches, sizeof(PVPatch*));
   assert(tmpPatch != NULL);
   //setKernelPatches(newKernelPatch, axonId);

   pvdata_t * data_patches = pvpatches_inplace_new(tmpPatch, nxPatch, nyPatch, nfPatch, numKernelPatches);

/*
   for (int kernelIndex = 0; kernelIndex < numKernelPatches; kernelIndex++) {
      //kernelPatches[axonId][kernelIndex] = pvpatch_inplace_new(nxPatch, nyPatch, nfPatch);
      tmpPatch[kernelIndex] = pvpatch_inplace_new(nxPatch, nyPatch, nfPatch);
      assert(tmpPatch[kernelIndex] != NULL );
   }
*/

   // need to allocate PVPatch for each pre-synaptic cell to store patch dimensions because patches may be shrunken
   for (int patchIndex = 0; patchIndex < nPatches; patchIndex++) {
      patches[axonId][patchIndex] = pvpatch_new(nxPatch, nyPatch, nfPatch);
   }


   for (int patchIndex = 0; patchIndex < nPatches; patchIndex++) {
      int kernelIndex = this->patchIndexToKernelIndex(patchIndex);
      patches[axonId][patchIndex]->data = tmpPatch[kernelIndex]->data;
   }
//   return patches[0]->data;
   return data_patches;
}

int KernelConn::initializeUpdateTime(PVParams * params) {
   if( plasticityFlag ) {
      float defaultUpdatePeriod = 1.f;
      weightUpdatePeriod = params->value(name, "weightUpdatePeriod", defaultUpdatePeriod);
   }
   return PV_SUCCESS;
}

int KernelConn::shrinkPatches(int arborId) {
   int numPatches = numDataPatches();
   for (int kex = 0; kex < numPatches; kex++) {
      HyPerConn::shrinkPatch(kernelIndexToPatchIndex(kex), arborId);
   } // loop over pre-synaptic neurons

   return 0;
}

/*TODO  createWeights currently breaks in this subclass if called more than once,
 * fix interface by adding extra dataPatches argument to overloaded method
 * so asserts are unnecessary
 */
/*
PVPatch ** KernelConn::createWeights(PVPatch ** patches, int nPatches, int nxPatch,
      int nyPatch, int nfPatch, int axonId)
{
   //assert(numberOfAxonalArborLists() == 1);

   assert(patches == NULL);

   patches = (PVPatch**) calloc(sizeof(PVPatch*), nPatches);
   assert(patches != NULL);

   //assert(kernelPatches == NULL);
   allocWeights(patches, nPatches, nxPatch, nyPatch, nfPatch, axonId);

   return patches;
}
*/

int KernelConn::deleteWeights()
{
   //const int arbor = 0;

   for(int n=0;n<numberOfAxonalArborLists(); n++) {
      for (int k = 0; k < numDataPatches(); k++) {
         pvpatch_inplace_delete(kernelPatches[n][k]);
      }
      free(kernelPatches[n]);
   }
   free(kernelPatches);

   return HyPerConn::deleteWeights();
}

PVPatch ***  KernelConn::initializeWeights(PVPatch *** arbors, int numPatches,
      const char * filename)
{
   //int arbor = 0;
   int numKernelPatches = numDataPatches();
   HyPerConn::initializeWeights(kernelPatches, numKernelPatches, filename);
   return arbors;
}

PVPatch ** KernelConn::readWeights(PVPatch ** patches, int numPatches,
      const char * filename)
{
   //HyPerConn::readWeights(patches, numPatches, filename);

   return patches;
}

int KernelConn::numDataPatches()
{
   int nxKernel = (pre->getXScale() < post->getXScale()) ? pow(2,
         post->getXScale() - pre->getXScale()) : 1;
   int nyKernel = (pre->getYScale() < post->getYScale()) ? pow(2,
         post->getYScale() - pre->getYScale()) : 1;
   int numKernelPatches = pre->clayer->loc.nf * nxKernel * nyKernel;
   return numKernelPatches;
}

float KernelConn::minWeight(int arborId)
{
   //const int axonID = 0;
   const int numKernels = numDataPatches();
   const int numWeights = nxp * nyp * nfp;
   float min_weight = FLT_MAX;
   for (int iKernel = 0; iKernel < numKernels; iKernel++) {
      pvdata_t * kernelWeights = kernelPatches[arborId][iKernel]->data;
      for (int iWeight = 0; iWeight < numWeights; iWeight++) {
         min_weight = (min_weight < kernelWeights[iWeight]) ? min_weight
               : kernelWeights[iWeight];
      }
   }
   return min_weight;
}

float KernelConn::maxWeight(int arborId)
{
   //const int axonID = 0;
   const int numKernels = numDataPatches();
   const int numWeights = nxp * nyp * nfp;
   float max_weight = -FLT_MAX;
   for (int iKernel = 0; iKernel < numKernels; iKernel++) {
      pvdata_t * kernelWeights = kernelPatches[arborId][iKernel]->data;
      for (int iWeight = 0; iWeight < numWeights; iWeight++) {
         max_weight = (max_weight > kernelWeights[iWeight]) ? max_weight
               : kernelWeights[iWeight];
      }
   }
   return max_weight;
}

int KernelConn::calc_dW(int axonId){
   clear_dW(axonId);
   update_dW(axonId);
   return PV_BREAK;
}

int KernelConn::clear_dW(int axonId) {
   // zero dWeightPatches
   for(int kAxon = 0; kAxon < this->numberOfAxonalArborLists(); kAxon++){
      for(int kKernel = 0; kKernel < this->numDataPatches(); kKernel++){
         PVPatch * dKernelPatch = dKernelPatches[kAxon][kKernel];
         //assert(dKernelPatch->sy == kernelPatches[kAxon][kKernel]->sy); // eventually dKernelPatches->data will be split out and the other fields will be replaced by references to kernelPatches[kAxon][kKernel]
         int syPatch = syp; //dKernelPatch->sy;
         int nkPatch = nfp * dKernelPatch->nx;
         float * dWeights = dKernelPatch->data;
         for(int kyPatch = 0; kyPatch < dKernelPatch->ny; kyPatch++){
            for(int kPatch = 0; kPatch < nkPatch; kPatch++){
               dWeights[kPatch] = 0.0f;
            }
            dWeights += syPatch;
         }
      }
   }
   return PV_SUCCESS;
}
int KernelConn::update_dW(int axonId) {
   // Typically override this method with a call to defaultUpdate_dW(axonId)
   return PV_SUCCESS;
}

int KernelConn::defaultUpdate_dW(int axonId) {
   // compute dW but don't add them to the weights yet.
   // That takes place in reduceKernels, so that the output is
   // independent of the number of processors.
   int nExt = preSynapticLayer()->getNumExtended();
   int numKernelIndices = numDataPatches();
   const pvdata_t * preactbuf = preSynapticLayer()->getLayerData(getDelay(axonId));
   const pvdata_t * postactbuf = postSynapticLayer()->getLayerData(getDelay(axonId));

   int sya = (post->getLayerLoc()->nf * (post->getLayerLoc()->nx + 2*post->getLayerLoc()->nb));
   // int syw = syp;
   for(int kExt=0; kExt<nExt;kExt++) {
      PVPatch * weights = getWeights(kExt,axonId);
      size_t offset = getAPostOffset(kExt, axonId);
      pvdata_t preact = preactbuf[kExt];
      int ny = weights->ny;
      int nk = weights->nx * nfp;
      const pvdata_t * postactRef = &postactbuf[offset];
      pvdata_t * dwdata = get_dWData(kExt, axonId);
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
      int numpatchitems = dKernelPatches[axonId][kernelindex]->nx * dKernelPatches[axonId][kernelindex]->ny * nfp;
      pvdata_t * dwpatchdata = dKernelPatches[axonId][kernelindex]->data;
      for( int n=0; n<numpatchitems; n++ ) {
         dwpatchdata[n] /= divisor;
      }
   }

   lastUpdateTime = parent->simulationTime();

   return PV_SUCCESS;
}

pvdata_t KernelConn::updateRule_dW(pvdata_t pre, pvdata_t post) {
   return pre*post;
}

int KernelConn::updateState(float time, float dt) {
   update_timer->start();
   int status = PV_SUCCESS;
   if( !plasticityFlag ) {
      return status;
   }
   if( time >= weightUpdateTime) {
      computeNewWeightUpdateTime(time, weightUpdateTime);
      for(int axonID=0;axonID<numberOfAxonalArborLists();axonID++) {
         status = calc_dW(axonID);  // calculate changes in weights
         if (status == PV_BREAK) {break;}
         assert(status == PV_SUCCESS);
      }

#ifdef PV_USE_MPI
      for(int axonID=0;axonID<numberOfAxonalArborLists();axonID++) {
         status = reduceKernels(axonID);  // combine partial changes in each column
         if (status == PV_BREAK) {break;}
         assert(status == PV_SUCCESS);
      }
#endif // PV_USE_MPI

      for(int axonID=0;axonID<numberOfAxonalArborLists();axonID++) {
         status = updateWeights(axonID);  // calculate new weights from changes
         if (status == PV_BREAK) {break;}
         assert(status == PV_SUCCESS);
      }
      if( normalize_flag ) {
         for(int axonID=0;axonID<numberOfAxonalArborLists();axonID++) {
            status = normalizeWeights(kernelPatches[axonID], numDataPatches(), axonID);
            if (status == PV_BREAK) {break;}
            assert(status == PV_SUCCESS);
         }
      }  // normalize_flag
   } // time > weightUpdateTime

update_timer->stop();
return PV_SUCCESS;
} // updateState

int KernelConn::updateWeights(int axonId){
   lastUpdateTime = parent->simulationTime();
   // add dKernelPatches to KernelPatches
   for(int kAxon = 0; kAxon < this->numberOfAxonalArborLists(); kAxon++){
      for(int kKernel = 0; kKernel < this->numDataPatches(); kKernel++){
         PVPatch * kernelPatch = kernelPatches[kAxon][kKernel];
         PVPatch * dKernelPatch = dKernelPatches[kAxon][kKernel];
         int syPatch = syp;
         int nkPatch = nfp * kernelPatch->nx;
         pvdata_t * weights = kernelPatch->data;
         pvdata_t * dWeights = dKernelPatch->data;
         for(int kyPatch = 0; kyPatch < kernelPatch->ny; kyPatch++){
            for(int kPatch = 0; kPatch < nkPatch; kPatch++){
               weights[kPatch] += dWeights[kPatch];
            }
            weights += syPatch;
            dWeights += syPatch;
         }
      }
   }
   return PV_BREAK;
}

float KernelConn::computeNewWeightUpdateTime(float time, float currentUpdateTime) {
   // Is only called by KernelConn::updateState if plasticityFlag is true
   weightUpdateTime += weightUpdatePeriod;
   return weightUpdateTime;
}

#ifdef PV_USE_MPI
int KernelConn::reduceKernels(const int axonID) {
   const int numPatches = numDataPatches();
   const size_t patchSize = nxp*nyp*nfp*sizeof(pvdata_t);
   const size_t localSize = numPatches * patchSize;

   //Now uses member variable mpiReductionBuffer
   // pvdata_t * buf = (pvdata_t *) malloc(localSize*sizeof(pvdata_t));
   // if(buf == NULL) {
   //    fprintf(stderr, "KernelConn::reduceKernels:Unable to allocate memory\n");
   //    exit(1);
   // }

   // Copy this column's weights into mpiReductionBuffer
   int idx = 0;
   for (int k = 0; k < numPatches; k++) {
      //PVPatch * p = kernelPatches[axonID][k];
      PVPatch * p = dKernelPatches[axonID][k];
      const pvdata_t * data = p->data;

      //const int sxp = p->sx;
      //const int syp = p->sy;
      //const int sfp = p->sf;

      for (int y = 0; y < p->ny; y++) {
         for (int x = 0; x < p->nx; x++) {
            for (int f = 0; f < nfp; f++) {
               mpiReductionBuffer[idx] = data[x*sxp + y*syp + f*sfp];
               idx++;
            }
         }
      }
   }

   // MPI_Allreduce combines all processors' buffers and puts the common result
   // into each processor's buffer.
   Communicator * comm = parent->icCommunicator();
   const MPI_Comm mpi_comm = comm->communicator();
   int ierr;
   ierr = MPI_Allreduce(MPI_IN_PLACE, mpiReductionBuffer, localSize, MPI_FLOAT, MPI_SUM, mpi_comm);
   // TODO error handling

   // mpiReductionBuffer now holds the sum over all processes.
   // Divide by number of processes to get average and copy back to patches
   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();
   const int nProcs = nxProcs * nyProcs;
   idx = 0;
   for (int k = 0; k < numPatches; k++) {
      //PVPatch * p = kernelPatches[axonID][k];
      PVPatch * p = dKernelPatches[axonID][k];
      pvdata_t * data = p->data;

      //const int sxp = p->sx;
      //const int syp = p->sy;
      //const int sfp = p->sf;

      for (int y = 0; y < p->ny; y++) {
         for (int x = 0; x < p->nx; x++) {
            for (int f = 0; f < nfp; f++) {
               data[x*sxp + y*syp + f*sfp] = mpiReductionBuffer[idx]/nProcs;
               idx++;
            }
         }
      }
   }

   // free(buf);
   return PV_SUCCESS;
}
#endif // PV_USE_MPI

int KernelConn::correctPIndex(int patchIndex) {
   return kernelIndexToPatchIndex(patchIndex);
}


int KernelConn::checkNormalizeArbor(PVPatch ** patches, int numPatches, int arborId)
{
   int status = PV_BREAK;
   const int num_kernels = numDataPatches();
   for (int kPatch = 0; kPatch < num_kernels; kPatch++) {
      double sumAll = 0.0f;
      double sum2All = 0.0f;
      float maxAll = 0.0f;
      for(int kArbor = 0; kArbor < this->numberOfAxonalArborLists(); kArbor++){
         double sum, sum2;
         float maxVal;
         status = sumWeights(kernelPatches[kArbor][kPatch], &sum, &sum2, &maxVal);
         assert( (status == PV_SUCCESS) || (status == PV_BREAK) );
         sumAll += sum;
         sum2All += sum2;
         maxAll = maxVal > maxAll ? maxVal : maxAll;
      } // kArbor
      int num_weights = nxp * nyp * nfp * numberOfAxonalArborLists();
      float sigma2 = ( sumAll / num_weights ) - ( sumAll / num_weights ) * ( sumAll / num_weights );
      for(int kArbor = 0; kArbor < this->numberOfAxonalArborLists(); kArbor++){
         if( sumAll != 0 || sigma2 != 0 ) {
            status = checkNormalizeWeights(kernelPatches[kArbor][kPatch], sumAll, sigma2, maxAll);
            assert(status == PV_SUCCESS );
         }
         else {
            fprintf(stderr, "checkNormalizeArbor: connection \"%s\", arbor %d, kernel %d has all zero weights.\n", name, kArbor, kPatch);
         }
      }
   }
   return PV_BREAK;
} // checkNormalizeArbor


int KernelConn::normalizeWeights(PVPatch ** patches, int numPatches, int arborId)
{
   int status = PV_SUCCESS;
   const int num_kernels = numDataPatches();

   // symmetrize before normalization
   if ( symmetrizeWeightsFlag ){
      for(int kArbor = 0; kArbor < this->numberOfAxonalArborLists(); kArbor++){
         status = symmetrizeWeights(kernelPatches[kArbor], num_kernels, kArbor);
         assert( (status == PV_SUCCESS) || (status == PV_BREAK) );
      }
   }

   // normalize after symmetrization
   if (this->normalize_arbors_individually) {
      for(int kArbor = 0; kArbor < this->numberOfAxonalArborLists(); kArbor++){
         status = HyPerConn::normalizeWeights(kernelPatches[kArbor], num_kernels, kArbor);
         assert( (status == PV_SUCCESS) || (status == PV_BREAK) );
      }
      status = PV_BREAK;
   }
   else {  // default behavior
      for (int kPatch = 0; kPatch < num_kernels; kPatch++) {
         double sumAll = 0.0f;
         double sum2All = 0.0f;
         float maxAll = 0.0f;
         for(int kArbor = 0; kArbor < this->numberOfAxonalArborLists(); kArbor++){
            double sum, sum2;
            float maxVal;
            status = sumWeights(kernelPatches[kArbor][kPatch], &sum, &sum2, &maxVal);
            assert( (status == PV_SUCCESS) || (status == PV_BREAK) );
            sumAll += sum;
            sum2All += sum2;
            maxAll = maxVal > maxAll ? maxVal : maxAll;
         } // kArbor
         for(int kArbor = 0; kArbor < this->numberOfAxonalArborLists(); kArbor++){
            status = scaleWeights(kernelPatches[kArbor][kPatch], sumAll, sum2All, maxAll);
            assert( (status == PV_SUCCESS) || (status == PV_BREAK) );
         }
      } // kPatch < numPatches

      status = checkNormalizeArbor(patches, numPatches, arborId);
      assert( (status == PV_SUCCESS) || (status == PV_BREAK) );
      status = PV_BREAK;
   } // this->normalize_arbors_individually
   return status;
}

int KernelConn::symmetrizeWeights(PVPatch ** patches, int numPatches, int arborId)
{
   int status = PV_SUCCESS;
   if (arborId == 0)
      printf("Entering KernelConn::symmetrizeWeights for connection \"%s\"\n", name);
   assert(pre->clayer->loc.nf==post->clayer->loc.nf);
   PVPatch ** symPatches;
   symPatches = (PVPatch**) calloc(sizeof(PVPatch*), numPatches);
   assert(symPatches != NULL);
   for (int iKernel = 0; iKernel < numPatches; iKernel++) {
      symPatches[iKernel] = pvpatch_inplace_new(nxp, nyp, nfp);
      assert(symPatches[iKernel] != NULL );
   }
   const int sy = nxp * nfp;
   const float deltaTheta = PI / nfp;
   const float offsetTheta = 0.5f * deltaTheta;
   const int kyMid = nyp / 2;
   const int kxMid = nxp / 2;
   for (int iSymKernel = 0; iSymKernel < numPatches; iSymKernel++) {
      PVPatch * symWp = symPatches[iSymKernel];
      pvdata_t * symW = symWp->data;
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
                  PVPatch * kerWp = getKernelPatch(arborId, iKernel);
                  pvdata_t * kerW = kerWp->data;
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
      PVPatch * kerWp = kernelPatches[arborId][iKernel];
      pvdata_t * kerW = kerWp->data;
      PVPatch * symWp = symPatches[iKernel];
      pvdata_t * symW = symWp->data;
      for (int iW = 0; iW < num_weights; iW++) {
         kerW[iW] = symW[iW];
      }
   } // iKernel
   for (int iKernel = 0; iKernel < numPatches; iKernel++) {
      pvpatch_inplace_delete(symPatches[iKernel]);
   } // iKernel
   free(symPatches);
   printf("Exiting KernelConn::symmetrizeWeights for connection \"%s\"\n", name);
   return status;
}

int KernelConn::writeWeights(float timef, bool last) {
   const int numPatches = numDataPatches();
   return HyPerConn::writeWeights(kernelPatches, numPatches, NULL, timef, last);
}

int KernelConn::writeWeights(const char * filename) {
   return HyPerConn::writeWeights(kernelPatches, numDataPatches(), filename, parent->simulationTime(), true);
}

#ifdef OBSOLETE_NBANDSFORARBORS
int KernelConn::writeWeights(float time, bool last)
{
   //const int arbor = 0;
   this->fileType = PVP_KERNEL_FILE_TYPE;
   const int numPatches = numDataPatches();
   for(int arborId=0;arborId<numberOfAxonalArborLists();arborId++) {
      if(HyPerConn::writeWeights(kernelPatches[arborId], numPatches, NULL, time, last, arborId))
         return 1;
   }
   return 0;
}
#endif // OBSOLETE_NBANDSFORARBORS

int KernelConn::checkpointRead(float * timef) {
   char * filename = checkpointFilename();
   InitWeights * weightsInitObject = new InitWeights();
   weightsInitObject->initializeWeights(kernelPatches, numDataPatches(), filename, this, timef);
   free(filename);
   return PV_SUCCESS;
}

int KernelConn::checkpointWrite() {
   char * filename;
   filename = (char *) malloc( (strlen(name)+12)*sizeof(char) );
   assert(filename != NULL);
   sprintf(filename, "%s_W.pvp", name);
   return HyPerConn::writeWeights(kernelPatches, numDataPatches(), filename, parent->simulationTime(), true);
}

// one to many mapping, chose first patch index in restricted space
// kernelIndex for unit cell
// patchIndex in extended space
int KernelConn::kernelIndexToPatchIndex(int kernelIndex, int * kxPatchIndex,
      int * kyPatchIndex, int * kfPatchIndex)
{
   int patchIndex;
   // get size of kernel PV cube
   int nxKernel_tmp = (pre->getXScale() < post->getXScale()) ? pow(2,
         post->getXScale() - pre->getXScale()) : 1;
   int nyKernel_tmp = (pre->getYScale() < post->getYScale()) ? pow(2,
         post->getYScale() - pre->getYScale()) : 1;
   int nfKernel_tmp = pre->getLayerLoc()->nf;
   int kxPreExtended = kxPos(kernelIndex, nxKernel_tmp, nyKernel_tmp, nfKernel_tmp) + pre->getLayerLoc()->nb;
   int kyPreExtended = kyPos(kernelIndex, nxKernel_tmp, nyKernel_tmp, nfKernel_tmp) + pre->getLayerLoc()->nb;
   int kfPre = featureIndex(kernelIndex, nxKernel_tmp, nyKernel_tmp, nfKernel_tmp);
   int nxPreExtended = pre->getLayerLoc()->nx + 2*pre->getLayerLoc()->nb;
   int nyPreExtended = pre->getLayerLoc()->ny + 2*pre->getLayerLoc()->nb;
   patchIndex = kIndex(kxPreExtended, kyPreExtended, kfPre, nxPreExtended, nyPreExtended, nfKernel_tmp);
   if (kxPatchIndex != NULL){
      *kxPatchIndex = kxPreExtended;
   }
   if (kyPatchIndex != NULL){
      *kyPatchIndex = kyPreExtended;
   }
   if (kfPatchIndex != NULL){
      *kfPatchIndex = kfPre;
   }
   return patchIndex;
}

// many to one mapping from weight patches to kernels
// patchIndex always in extended space
// kernelIndex always for unit cell
int KernelConn::patchIndexToKernelIndex(int patchIndex, int * kxKernelIndex,
      int * kyKernelIndex, int * kfKernelIndex)
{
   int kernelIndex;
   int nxPreExtended = pre->getLayerLoc()->nx + 2*pre->getLayerLoc()->nb;
   int nyPreExtended = pre->getLayerLoc()->ny + 2*pre->getLayerLoc()->nb;
   int nfPre = pre->getLayerLoc()->nf;
   int kxPreExtended = kxPos(patchIndex, nxPreExtended, nyPreExtended, nfPre);
   int kyPreExtended = kyPos(patchIndex, nxPreExtended, nyPreExtended, nfPre);

   // check that patchIndex lay within margins
   assert(kxPreExtended >= 0);
   assert(kyPreExtended >= 0);
   assert(kxPreExtended < nxPreExtended);
   assert(kyPreExtended < nyPreExtended);

   // convert from extended to restricted space (in local HyPerCol coordinates)
   int kxPreRestricted;
   kxPreRestricted = kxPreExtended - pre->getLayerLoc()->nb;
   while(kxPreRestricted < 0){
      kxPreRestricted += pre->getLayerLoc()->nx;
   }
   while(kxPreRestricted >= pre->getLayerLoc()->nx){
      kxPreRestricted -= pre->getLayerLoc()->nx;
   }

   int kyPreRestricted;
   kyPreRestricted = kyPreExtended - pre->getLayerLoc()->nb;
   while(kyPreRestricted < 0){
      kyPreRestricted += pre->getLayerLoc()->ny;
   }
   while(kyPreRestricted >= pre->getLayerLoc()->ny){
      kyPreRestricted -= pre->getLayerLoc()->ny;
   }

   int kfPre = featureIndex(patchIndex, nxPreExtended, nyPreExtended, nfPre);

   int nxKernel_tmp = (pre->getXScale() < post->getXScale()) ? pow(2,
         post->getXScale() - pre->getXScale()) : 1;
   int nyKernel_tmp = (pre->getYScale() < post->getYScale()) ? pow(2,
         post->getYScale() - pre->getYScale()) : 1;
   int kxKernel = kxPreRestricted % nxKernel_tmp;
   int kyKernel = kyPreRestricted % nyKernel_tmp;

   kernelIndex = kIndex(kxKernel, kyKernel, kfPre, nxKernel_tmp, nyKernel_tmp, nfPre);
   if (kxKernelIndex != NULL){
      *kxKernelIndex = kxKernel;
   }
   if (kyKernelIndex != NULL){
      *kyKernelIndex = kyKernel;
   }
   if (kfKernelIndex != NULL){
      *kfKernelIndex = kfPre;
   }
   return kernelIndex;
}


} // namespace PV

