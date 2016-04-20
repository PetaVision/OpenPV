/*
 * ImprintConn.cpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#include "ImprintConn.hpp"
#include <iostream>

namespace PV {

ImprintConn::ImprintConn(){
   initialize_base();
}

ImprintConn::ImprintConn(const char * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer) : HyPerConn()
{
   initialize_base();
   initialize(name, hc, weightInitializer, weightNormalizer);
}

ImprintConn::~ImprintConn() {
   free(lastActiveTime);
   free(imprinted);
}

int ImprintConn::initialize_base() {
   imprintTimeThresh = -1;
   imprintChance = .2;
   return PV_SUCCESS;
}

int ImprintConn::allocateDataStructures() {
   int status = HyPerConn::allocateDataStructures();
   if (status==PV_POSTPONE) { return status; }
   const PVLayerLoc * loc = pre->getLayerLoc();
   int numKernelIndices = getNumDataPatches();
   imprinted = (bool*) calloc(numKernelIndices * numberOfAxonalArborLists(), sizeof(bool));
   lastActiveTime = (double*) malloc(numKernelIndices * numberOfAxonalArborLists() * sizeof(double));
   for(int ki = 0; ki < numKernelIndices * numberOfAxonalArborLists(); ki++){
      //Do we start something other than 0?
      lastActiveTime[ki] = ki;
   }
   return status;
}

int ImprintConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   ioParam_imprintTimeThresh(ioFlag);
   //ioParam_imprintChance(ioFlag);
   return status;
}

// TODO: make sure code works in non-shared weight case
void ImprintConn::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   sharedWeights = true;
   if (ioFlag == PARAMS_IO_READ) {
      fileType = PVP_KERNEL_FILE_TYPE;
      parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights", true/*correctValue*/);
   }
}

void ImprintConn::ioParam_imprintTimeThresh(enum ParamsIOFlag ioFlag) {
   parent->ioParamValueRequired(ioFlag, name, "imprintTimeThresh", &imprintTimeThresh);
   if (ioFlag==PARAMS_IO_READ) {
      if (imprintTimeThresh==-1) {
         imprintTimeThresh = weightUpdateTime * 100; //Default value of 100 weight updates
      }
      else if(imprintTimeThresh <= weightUpdateTime && parent->columnId()==0){
         fprintf(stderr, "Warning: ImprintConn's imprintTimeThresh is smaller than weightUpdateTime. The algorithm will imprint on every weight update\n");
      }
   }
}

//void ImprintConn::ioParam_imprintChance(enum ParamsIOFlag ioFlag) {
//   parent->ioParamValue(ioFlag, name, "imprintChance", &imprintChance, imprintChance);
//   if (ioFlag==PARAMS_IO_READ) {
//      if (imprintTimeThresh==-1) {
//         imprintTimeThresh = weightUpdateTime * 100; //Default value of 100 weight updates
//      }
//      else if(imprintTimeThresh <= weightUpdateTime && parent->columnId()==0){
//         fprintf(stderr, "Warning: ImprintConn's imprintTimeThresh is smaller than weightUpdateTime. The algorithm will imprint on every weight update\n");
//      }
//   }
//}

int ImprintConn::imprintFeature(int arbor_ID, int batch_ID, int kExt){
   const pvdata_t * postactbufAll = postSynapticLayer()->getLayerData(); 
   const pvdata_t * postactbuf = postactbufAll + batch_ID * postSynapticLayer()->getNumExtended();

   const PVLayerLoc * postLoc = post->getLayerLoc();

   PVPatch * weights = getWeights(kExt,arbor_ID);

   int sya = (post->getLayerLoc()->nf * (post->getLayerLoc()->nx + post->getLayerLoc()->halo.lt + post->getLayerLoc()->halo.rt));

   size_t offset = getAPostOffset(kExt, arbor_ID);
   const pvdata_t * postactRef = &postactbuf[offset];

   int sym = 0;
   const pvdata_t * maskactRef = NULL;
   if(useMask){
      const pvdata_t * maskactbufAll = mask->getLayerData();
      const pvdata_t * maskactbuf = maskactbufAll + batch_ID * mask->getNumExtended();
      maskactRef = &maskactbuf[offset];
      sym = (mask->getLayerLoc()->nf * (mask->getLayerLoc()->nx + mask->getLayerLoc()->halo.lt + mask->getLayerLoc()->halo.rt));
   }

   int ny = weights->ny;
   int nk = weights->nx * nfp;
   int kernelIndex = patchIndexToDataIndex(kExt);


   pvwdata_t * dwdata = get_dwData(arbor_ID, kExt);
   long * activations = NULL;
   if(normalizeDwFlag){
      activations = get_activations(arbor_ID, kExt);
   }
   
   int lineoffsetw = 0;
   int lineoffseta = 0;
   int lineoffsetm = 0;

   for( int y=0; y<ny; y++ ) {
      for( int k=0; k<nk; k++ ) {
         pvdata_t aPost = postactRef[lineoffseta+k];
         //calculate contribution to dw unless masked out
         assert(!useMask || maskactRef!=NULL); // if useMask is true, maskactRef must not be null
         float maskVal = 1;
         if(useMask){
            if(mask->getLayerLoc()->nf == 1){
               maskVal = maskactRef[lineoffsetm+((int)k/postLoc->nf)];
            }
            else{
               maskVal = maskactRef[lineoffsetm+k];
            }
         }
         if (maskVal != 0){
            assert(sharedWeights);
            //Offset in the case of a shrunken patch, where dwdata is applying when calling get_dwData
            if(normalizeDwFlag){
               activations[lineoffsetw + k]++;
            }
            //Set actual values to dwData. The imprinted buffer will tell updateWeights to update this kernel by setting to dwWeight
            dwdata[lineoffsetw + k] += aPost;
         }
      }
      lineoffsetw += syp;
      lineoffseta += sya;
      lineoffsetm += sym;
   }
   return PV_SUCCESS;
}
int ImprintConn::initialize_dW(int arborId){
   HyPerConn::initialize_dW(arborId);
   for(int ki = 0; ki < numberOfAxonalArborLists() * getNumDataPatches(); ki++){
      imprinted[ki] = false;
   }
   return PV_BREAK;
}

int ImprintConn::update_dW(int arbor_ID){
   // That takes place in reduceKernels, so that the output is
   // independent of the number of processors.
   int nExt = preSynapticLayer()->getNumExtended();
   int numKernelIndices = getNumDataPatches();
   const PVLayerLoc * loc = pre->getLayerLoc();
   const pvdata_t * preactbuf = preSynapticLayer()->getLayerData(getDelay(arbor_ID));
   int arborStart = arbor_ID * numKernelIndices;
   int patchSize = nxp * nyp * nfp;
   int nbatch = loc->nbatch;

   //Calculate x and y cell size
   int xCellSize = zUnitCellSize(pre->getXScale(), post->getXScale());
   int yCellSize = zUnitCellSize(pre->getYScale(), post->getYScale());
   int nxExt = loc->nx + loc->halo.lt + loc->halo.rt;
   int nyExt = loc->ny + loc->halo.up + loc->halo.dn;
   int nf = loc->nf;
   int numKernels = getNumDataPatches();

   for(int b = 0; b < nbatch; b++){
      //Shared weights done in parallel, parallel in numkernels
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for(int kernelIdx = 0; kernelIdx < numKernels; kernelIdx++){
         //Calculate xCellIdx, yCellIdx, and fCellIdx from kernelIndex
         int kxCellIdx = kxPos(kernelIdx, xCellSize, yCellSize, nf);
         int kyCellIdx = kyPos(kernelIdx, xCellSize, yCellSize, nf);
         int kfIdx = featureIndex(kernelIdx, xCellSize, yCellSize, nf);

         if (parent->simulationTime() - lastActiveTime[arborStart + kernelIdx] > imprintTimeThresh){
            imprinted[arborStart + kernelIdx] = true;
            lastActiveTime[arborStart + kernelIdx] = parent->simulationTime();
            std::cout << "Imprinted feature: Arbor " << arbor_ID << " kernel " << kernelIdx << "\n";
         }

         //Loop over all cells in pre ext
         int kyIdx = kyCellIdx;
         int yCellIdx = 0;
         while(kyIdx < nyExt){
            int kxIdx = kxCellIdx;
            int xCellIdx = 0;
            while(kxIdx < nxExt){
               //Calculate kExt from ky, kx, and kf
               int kExt = kIndex(kxIdx, kyIdx, kfIdx, nxExt, nyExt, nf);
               if(imprinted[arborStart + kernelIdx]){
                  imprintFeature(arbor_ID, b, kExt);
               }
               else{
                  int status = updateInd_dW(arbor_ID, b, kExt);
                  //Status will be PV_CONTINUE if preact is 0 (not active)
                  if(status == PV_SUCCESS){
                     lastActiveTime[arborStart + kernelIdx] = parent->simulationTime();
                  }
               }
               xCellIdx++;
               kxIdx = kxCellIdx + xCellIdx * xCellSize;
            }
            yCellIdx++;
            kyIdx = kyCellIdx + yCellIdx * yCellSize;
         }
      }

      //If update from clones, update dw here as well
      //Updates on all PlasticClones
      for(int clonei = 0; clonei < clones.size(); clonei++){
         assert(clones[clonei]->preSynapticLayer()->getNumExtended() == nExt);
         for(int kExt=0; kExt<nExt;kExt++) {
            int kernelIndex = clones[clonei]->patchIndexToDataIndex(kExt);
            if(!imprinted[arborStart + kernelIndex]){
               clones[clonei]->updateInd_dW(arbor_ID, b, kExt);
            }
         }
      }

   }
   //Do mpi to update lastActiveTime
   int ierr = MPI_Allreduce(MPI_IN_PLACE, lastActiveTime, numKernelIndices * numberOfAxonalArborLists(), MPI_DOUBLE, MPI_MAX, parent->icCommunicator()->communicator());

   return PV_SUCCESS;
}

int ImprintConn::updateWeights(int arbor_ID){
   int numKernelIndices = getNumDataPatches();
   for(int kArbor = 0; kArbor < this->numberOfAxonalArborLists(); kArbor++){
      int arborStart = kArbor * numKernelIndices;
      pvwdata_t * w_data_start = get_wDataStart(kArbor);
      for(int kKernel = 0; kKernel < numKernelIndices; kKernel++){
         //Regular weight update
         for(int kPatch = 0; kPatch < nxp*nyp*nfp; kPatch++){
            int k = kKernel * nxp*nyp*nfp + kPatch;
            //If imprinte, dw buffer is set to post
            if(imprinted[arborStart + kKernel]){
               w_data_start[k] = get_dwDataStart(kArbor)[k];
            }
            //Otherwise, accumulate
            else{
               w_data_start[k] += get_dwDataStart(kArbor)[k];
            }
         }
      }
   }
   return PV_BREAK;
}

int ImprintConn::checkpointRead(const char * cpDir, double * timeptr) {
   int status = HyPerConn::checkpointRead(cpDir, timeptr);
   long numBuf = getNumDataPatches() * numberOfAxonalArborLists();
   if( parent->columnId() == 0 ) {
      char * filename = parent->pathInCheckpoint(cpDir, getName(), "_ImprintState.bin");
      PV_Stream * pvstream = PV_fopen(filename, "r", false/*verifyWrites*/);
      if( pvstream != NULL ) {
         status = PV_fread(lastActiveTime, sizeof(double), numBuf, pvstream) == 1 ? status : PV_FAILURE;
         PV_fclose(pvstream);
      }
      else {
         fprintf(stderr, "Unable to read from \"%s\"\n", filename);
         exit(-1);
      }
      free(filename);
   }

   if (parent->icCommunicator()->commSize()>1) {
      //Communicate buffer size to rest of processes
      MPI_Bcast(lastActiveTime, numBuf, MPI_DOUBLE, 0, parent->icCommunicator()->communicator());
   }
   return status;
}

int ImprintConn::checkpointWrite(const char * cpDir) {
   int status = HyPerConn::checkpointWrite(cpDir);
   long numBuf = getNumDataPatches() * numberOfAxonalArborLists();
   if( parent->columnId() == 0 ) {
      int filenamesize = strlen(cpDir)+1+strlen(name)+18;
      // The +1 is for the slash between cpDir and name; the +18 needs to be large enough to hold the suffix _PatternState.{bin,txt} plus the null terminator
      char * filename = (char *) malloc( filenamesize*sizeof(char) );
      assert(filename != NULL);
      sprintf(filename, "%s/%s_ImprintState.bin", cpDir, name);
      std::cout << "filename: " << filename << "\n";
      PV_Stream * pvstream = PV_fopen(filename, "w", parent->getVerifyWrites());
      if( pvstream != NULL ) {
         status = PV_fwrite(lastActiveTime, sizeof(double), numBuf, pvstream) == numBuf ? status : PV_FAILURE;
         PV_fclose(pvstream);
      }
      else {
         fprintf(stderr, "Unable to write to \"%s\"\n", filename);
         exit(-1);
      }
      if (status != PV_SUCCESS) {
         fprintf(stderr, "Patterns::checkpointWrite error: %s \"%s\" failed writing to %s\n", this->getKeyword(), name, filename);
         exit(EXIT_FAILURE);
      }
      free(filename); filename=NULL;
   }
   return PV_SUCCESS;
}

BaseObject * createImprintConn(char const * name, HyPerCol * hc) {
   if (hc==NULL) { return NULL; }
   InitWeights * weightInitializer = getWeightInitializer(name, hc);
   NormalizeBase * weightNormalizer = getWeightNormalizer(name, hc);
   return new ImprintConn(name, hc, weightInitializer, weightNormalizer);
}

} // end namespace PV
