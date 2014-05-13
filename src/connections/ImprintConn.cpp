/*
 * ImprintConn.cpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#include "ImprintConn.hpp"
namespace PV {

ImprintConn::ImprintConn(){
   initialize_base();
}

ImprintConn::ImprintConn(const char * name, HyPerCol * hc) : KernelConn()
{
   initialize_base();
   initialize(name, hc);
}

ImprintConn::~ImprintConn() {
   free(lastActiveTime);
   free(imprinted);
}

int ImprintConn::initialize_base() {
   imprintTimeThresh = -1;
   return PV_SUCCESS;
}

int ImprintConn::allocateDataStructures() {
   int status = KernelConn::allocateDataStructures();
   const PVLayerLoc * loc = pre->getLayerLoc();
   int numKernelIndices = getNumDataPatches();
   imprinted = (bool*) calloc(numKernelIndices, sizeof(bool));
   lastActiveTime = (double*) malloc(numKernelIndices * sizeof(double));
   for(int ki = 0; ki < numKernelIndices; ki++){
      lastActiveTime[ki] = ki * weightUpdatePeriod;
   }
   return status;
}

int ImprintConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = KernelConn::ioParamsFillGroup(ioFlag);
   ioParam_imprintTimeThresh(ioFlag);
   return status;
}

void ImprintConn::ioParam_imprintTimeThresh(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "imprintTimeThresh", &imprintTimeThresh, imprintTimeThresh);
   if (ioFlag==PARAMS_IO_READ) {
      if (imprintTimeThresh==-1) {
         imprintTimeThresh = weightUpdateTime * 100; //Default value of 100 weight updates
      }
      else if(imprintTimeThresh <= weightUpdateTime && parent->columnId()==0){
         fprintf(stderr, "Warning: ImprintConn's imprintTimeThresh is smaller than weightUpdateTime. The algorithm will imprint on every weight update\n");
      }
   }
}

bool ImprintConn::imprintFeature(int arborId, int kExt){
   PVPatch * weights = getWeights(kExt,arborId);
   //If imprinting, patch must not be a shrunken patch
   if(weights->nx != nxp || weights->ny != nyp){
      return false;
   }
   //Only one processor doing imprinting
   //Making a synced random number based off of sim time and kExt
   long syncedRandNum = (parent->simulationTime()/weightUpdatePeriod)+kExt;
   if(syncedRandNum % parent->icCommunicator()->commSize() != parent->icCommunicator()->commRank()){  
      return false;
   }

   int sya = (post->getLayerLoc()->nf * (post->getLayerLoc()->nx + 2*post->getLayerLoc()->nb));

   const pvdata_t * preactbuf = preSynapticLayer()->getLayerData(getDelay(arborId));
   const pvdata_t * postactbuf = postSynapticLayer()->getLayerData(); 
   size_t offset = getAPostOffset(kExt, arborId);
   int ny = weights->ny;
   int nk = weights->nx * nfp;
   const pvdata_t * postactRef = &postactbuf[offset];
   pvwdata_t * dwdata = get_dwData(arborId, kExt);
   int lineoffsetw = 0;
   int lineoffseta = 0;

   bool hasTexture = false;
   for( int y=0; y<ny; y++ ) {
      for( int k=0; k<nk; k++ ) {
          pvdata_t aPost = postactRef[lineoffseta+k];
          if(aPost != 0){
             hasTexture = true;
          }
          //Multiply aPost by a big number so it overpowers whatever is on the weights already
          //For some reason, this needs a huge number to make a difference
          //normalizing should take care of the scale after
          dwdata[lineoffsetw + k] = 9999999999 * aPost;
      }
      lineoffsetw += syp;
      lineoffseta += sya;
   }
   if(hasTexture){
      return true;
   }
   //If there's no texture in whatever was grabbed
   //Reset dwdata to 0
   for( int y=0; y<ny; y++ ) {
      for( int k=0; k<nk; k++ ) {
          dwdata[lineoffsetw + k] = 0;
      }
      lineoffsetw += syp;
      lineoffseta += sya;
   }
   return false;
}


int ImprintConn::update_dW(int arbor_ID){
   // compute dW but don't add them to the weights yet.
   // That takes place in reduceKernels, so that the output is
   // independent of the number of processors.
   int nExt = preSynapticLayer()->getNumExtended();
   int numKernelIndices = getNumDataPatches();
   const PVLayerLoc * preLoc = pre->getLayerLoc();
   const pvdata_t * preactbuf = preSynapticLayer()->getLayerData(getDelay(arbor_ID));

   //Reset numKernelActivations
   for(int ki = 0; ki < numKernelIndices; ki++){
      numKernelActivations[ki] = 0;
      imprinted[ki] = false;
   }

   for(int kExt=0; kExt<nExt;kExt++) {
      pvdata_t preact = preactbuf[kExt];
      //Check imprinting
      int kernelIndex = patchIndexToDataIndex(kExt);

      if (parent->simulationTime() - lastActiveTime[kernelIndex] > imprintTimeThresh){
         if(!imprinted[kernelIndex]){
            //Random chance (one in 5) to imprint
            if(rand() % 5 == 0){
               imprinted[kernelIndex] = imprintFeature(arbor_ID, kExt);
            }
            if(imprinted[kernelIndex]){
               std::cout << "Imprinted feature " << kernelIndex << "\n";
            }
         }
      }
      //Default update rule
      int status = defaultUpdateInd_dW(arbor_ID, kExt);
      if(status == PV_SUCCESS){
         lastActiveTime[kernelIndex] = parent->simulationTime();
      }
   }

   //Do mpi to update lastActiveTime
#ifdef PV_USE_MPI
   int ierr = MPI_Allreduce(MPI_IN_PLACE, lastActiveTime, numKernelIndices, MPI_DOUBLE, MPI_MAX, parent->icCommunicator()->communicator());
#endif

   normalize_dW(arbor_ID);

   lastUpdateTime = parent->simulationTime();

   return PV_SUCCESS;
}




} // end namespace PV
