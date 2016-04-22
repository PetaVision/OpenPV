/*
 * MomentumConn.cpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#include "MomentumConn.hpp"
#include <cstring>
#include "utils/PVAlloc.hpp"

namespace PV {

MomentumConn::MomentumConn(){
   initialize_base();
}

MomentumConn::MomentumConn(const char * name, HyPerCol * hc, InitWeights * weightInitializer, NormalizeBase * weightNormalizer) : HyPerConn()
{
   initialize_base();
   initialize(name, hc, weightInitializer, weightNormalizer);
}

MomentumConn::~MomentumConn() {
   if(momentumMethod){
      free(momentumMethod);
   }
   if(prev_dwDataStart) {
      free(prev_dwDataStart[0]);
      free(prev_dwDataStart);
   }
}

int MomentumConn::initialize_base() {
   prev_dwDataStart = NULL;
   momentumTau = .25;
   momentumMethod = NULL;
   momentumDecay = 0;
   timeBatchPeriod = 1;
   timeBatchIdx = -1;
   return PV_SUCCESS;
}

int MomentumConn::allocateDataStructures(){
   int status = HyPerConn::allocateDataStructures();
   if (status==PV_POSTPONE) { return status; }
   if (!plasticityFlag) return status;
   int sx = nfp;
   int sy = sx * nxp;
   int sp = sy * nyp;
   int nPatches = getNumDataPatches();

   const int numAxons = numberOfAxonalArborLists();

   //Allocate dw buffer for previous dw
   prev_dwDataStart = (pvwdata_t **) pvCalloc(numAxons, sizeof(pvwdata_t *));
   prev_dwDataStart[0] = (pvwdata_t*) pvCalloc(numAxons * nxp * nyp * nfp * nPatches, sizeof(pvwdata_t));
   for (int arborId = 0; arborId < numAxons; arborId++) {
      prev_dwDataStart[arborId] = (prev_dwDataStart[0] + sp * nPatches * arborId);
      assert(prev_dwDataStart[arborId] != NULL);
   } // loop over arbors

   //assert(clones.size() == 0);

   return PV_SUCCESS;
}

int MomentumConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   ioParam_momentumMethod(ioFlag);
   ioParam_momentumTau(ioFlag);
   ioParam_momentumDecay(ioFlag);
   ioParam_batchPeriod(ioFlag);
   return status;
}

void MomentumConn::ioParam_momentumTau(enum ParamsIOFlag ioFlag){
   if(plasticityFlag){
      float defaultVal = 0;
      if(strcmp(momentumMethod, "simple") == 0){
         defaultVal = .25;
      }
      else if(strcmp(momentumMethod, "viscosity") == 0){
         defaultVal = 100;
      }
      else if(strcmp(momentumMethod, "alex") == 0){
         defaultVal = .9;
      }
      
      parent->ioParamValue(ioFlag, name, "momentumTau", &momentumTau, defaultVal);
   }
}

/**
 * @brief momentumMethod: The momentum method to use
 * @details Assuming a = dwMax * pre * post
 * simple: deltaW(t) = a + momentumTau * deltaW(t-1)
 * viscosity: deltaW(t) = (deltaW(t-1) * exp(-1/momentumTau)) + a
 * alex: deltaW(t) = momentumTau * delta(t-1) - momentumDecay * dwMax * w(t) - a
 */
void MomentumConn::ioParam_momentumMethod(enum ParamsIOFlag ioFlag){
   if(plasticityFlag){
      parent->ioParamStringRequired(ioFlag, name, "momentumMethod", &momentumMethod);
      if(strcmp(momentumMethod, "simple") != 0 &&
         strcmp(momentumMethod, "viscosity") != 0 &&
         strcmp(momentumMethod, "alex")){
         std::cout << "MomentumConn " << name << ": momentumMethod of " << momentumMethod << " is not known, options are \"simple\", \"viscosity\", and \"alex\"\n";
         exit(-1);
      }
   }
}

void MomentumConn::ioParam_momentumDecay(enum ParamsIOFlag ioFlag){
   if(plasticityFlag){
      parent->ioParamValue(ioFlag, name, "momentumDecay", &momentumDecay, momentumDecay);
      if(momentumDecay < 0 || momentumDecay > 1){
         std::cout << "MomentumConn " << name << ": momentumDecay must be between 0 and 1 inclusive\n";
         exit(-1);
      }
   }
}

void MomentumConn::ioParam_batchPeriod(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if(plasticityFlag){
      parent->ioParamValue(ioFlag, name, "batchPeriod", &timeBatchPeriod, timeBatchPeriod);
   }
}


////Reduce kerenls should never be getting called except for checkpoint write, which is being overwritten.
//int MomentumConn::reduceKernels(const int arborID){
//   fprintf(stderr, "Error:  BatchConn calling reduceKernels\n");
//   exit(1);
//}
//
////No normalize reduction here, just summing up
//int MomentumConn::sumKernels(const int arborID) {
//   assert(sharedWeights && plasticityFlag);
//   Communicator * comm = parent->icCommunicator();
//   const MPI_Comm mpi_comm = comm->communicator();
//   int ierr;
//   const int nxProcs = comm->numCommColumns();
//   const int nyProcs = comm->numCommRows();
//   const int nProcs = nxProcs * nyProcs;
//   if (nProcs == 1){
//      //normalize_dW(arborID);
//      return PV_BREAK;
//   }
//   const int numPatches = getNumDataPatches();
//   const size_t patchSize = nxp*nyp*nfp;
//   const size_t localSize = numPatches * patchSize;
//   const size_t arborSize = localSize * this->numberOfAxonalArborLists();
//
//   ierr = MPI_Allreduce(MPI_IN_PLACE, this->get_dwDataStart(arborID), arborSize, MPI_FLOAT, MPI_SUM, mpi_comm);
//   return PV_BREAK;
//}
//
////Copied from HyPerConn
////Removed clearing of numKernelActivations, as it's done in updateState
//int MomentumConn::defaultUpdate_dW(int arbor_ID) {
//   int nExt = preSynapticLayer()->getNumExtended();
//   const PVLayerLoc * loc = preSynapticLayer()->getLayerLoc();
//
//   //Calculate x and y cell size
//   int xCellSize = zUnitCellSize(pre->getXScale(), post->getXScale());
//   int yCellSize = zUnitCellSize(pre->getYScale(), post->getYScale());
//   int nxExt = loc->nx + loc->halo.lt + loc->halo.rt;
//   int nyExt = loc->ny + loc->halo.up + loc->halo.dn;
//   int nf = loc->nf;
//   int numKernels = getNumDataPatches();
//
//   if(sharedWeights){
//      //Shared weights done in parallel, parallel in numkernels
//#ifdef PV_USE_OPENMP_THREADS
//#pragma omp parallel for
//#endif
//      for(int kernelIdx = 0; kernelIdx < numKernels; kernelIdx++){
//         //Calculate xCellIdx, yCellIdx, and fCellIdx from kernelIndex
//         int kxCellIdx = kxPos(kernelIdx, xCellSize, yCellSize, nf);
//         int kyCellIdx = kyPos(kernelIdx, xCellSize, yCellSize, nf);
//         int kfIdx = featureIndex(kernelIdx, xCellSize, yCellSize, nf);
//         //Loop over all cells in pre ext
//         int kyIdx = kyCellIdx;
//         int yCellIdx = 0;
//         while(kyIdx < nyExt){
//            int kxIdx = kxCellIdx;
//            int xCellIdx = 0;
//            while(kxIdx < nxExt){
//               //Calculate kExt from ky, kx, and kf
//               int kExt = kIndex(kxIdx, kyIdx, kfIdx, nxExt, nyExt, nf);
//               defaultUpdateInd_dW(arbor_ID, kExt);
//               xCellIdx++;
//               kxIdx = kxCellIdx + xCellIdx * xCellSize;
//            }
//            yCellIdx++;
//            kyIdx = kyCellIdx + yCellIdx * yCellSize;
//         }
//      }
//   }
//   else{
//      //No clobbering for non-shared weights
//#ifdef PV_USE_OPENMP_THREADS
//#pragma omp parallel for
//#endif
//      for(int kExt=0; kExt<nExt;kExt++) {
//         defaultUpdateInd_dW(arbor_ID, kExt);
//      }
//   }
//
//   return PV_SUCCESS;
//}

int MomentumConn::calc_dW() {
   assert(plasticityFlag);
   int status;
   timeBatchIdx = (timeBatchIdx + 1) % timeBatchPeriod;

   //Clear at time 0, update at time timeBatchPeriod - 1
   bool need_update_w = false;
   bool need_clear_dw = false;
   if(timeBatchIdx == 0){
      need_clear_dw = true;
   }

   //If updating next timestep, update weights here
   if((timeBatchIdx + 1) % timeBatchPeriod == 0){
      need_update_w = true;
   }

   for(int arborId=0;arborId<numberOfAxonalArborLists();arborId++) {
      //Clear every batch period
      if(need_clear_dw){
         status = initialize_dW(arborId);
         if (status==PV_BREAK) { break; }
         assert(status == PV_SUCCESS);
      }
   }

   for(int arborId=0;arborId<numberOfAxonalArborLists();arborId++) {
      //Sum up parts every timestep
      status = update_dW(arborId);
      if (status==PV_BREAK) { break; }
      assert(status == PV_SUCCESS);
   }

   for(int arborId=0;arborId<numberOfAxonalArborLists();arborId++) {
      //Reduce only when we need to update
      if(need_update_w){
         status = reduce_dW(arborId);
         if (status==PV_BREAK) { break; }
         assert(status == PV_SUCCESS);
      }
   }

   for(int arborId=0;arborId<numberOfAxonalArborLists();arborId++) {
      //Normalize only when reduced
      if(need_update_w){
         status = normalize_dW(arborId);
         if (status==PV_BREAK) { break; }
         assert(status == PV_SUCCESS);
      }
   }
   return PV_SUCCESS;
}

int MomentumConn::updateWeights(int arborId){
   if(timeBatchIdx != timeBatchPeriod - 1){
      return PV_SUCCESS;
   }
   //Add momentum right before updateWeights
   for(int kArbor = 0; kArbor < this->numberOfAxonalArborLists(); kArbor++){
      applyMomentum(arborId);
   }

   //Saved to prevweights
   assert(prev_dwDataStart);
   std::memcpy(*prev_dwDataStart, *get_dwDataStart(),
         sizeof(pvwdata_t) *
         numberOfAxonalArborLists() * 
         nxp * nyp * nfp *
         getNumDataPatches());


   // add dw to w
   for(int kArbor = 0; kArbor < this->numberOfAxonalArborLists(); kArbor++){
      pvwdata_t * w_data_start = get_wDataStart(kArbor);
      for( long int k=0; k<patchStartIndex(getNumDataPatches()); k++ ) {
         w_data_start[k] += get_dwDataStart(kArbor)[k];
      }
   }
   return PV_BREAK;
}

////Copied from HyPerConn, only change is a memcpy from dwweights to prev_dwweights
//int MomentumConn::updateState(double time, double dt){
//   int status = PV_SUCCESS;
//   if( !plasticityFlag ) {
//      return status;
//   }
//   update_timer->start();
//
//   //if (!combine_dW_with_W_flag) { clear_dW(); }
//   for(int arborId=0;arborId<numberOfAxonalArborLists();arborId++) {
//      status = calc_dW(arborId);        // Calculate changes in weights
//      if (status==PV_BREAK) { break; }
//      assert(status == PV_SUCCESS);
//   }
//
//   if(batchIdx >= batchPeriod - 1){
//      for (int arborID = 0; arborID < numberOfAxonalArborLists(); arborID++) {
//         sumKernels(arborID); // combine partial changes in each column
//         normalize_dW(arborID);
//         assert(status == PV_SUCCESS);
//      }
//
//      //Apply momentum
//      for (int arborID = 0; arborID < numberOfAxonalArborLists(); arborID++) {
//         status = applyMomentum(arborID);
//         if (status == PV_BREAK) {
//            break;
//         }
//         assert(status == PV_SUCCESS);
//      }
//      
//      //Update prev_dwData buffer
//      assert(prev_dwDataStart);
//      //After reduce, copy over to prev_dwData
//      std::memcpy(*prev_dwDataStart, *get_dwDataStart(),
//            sizeof(pvwdata_t) *
//            numberOfAxonalArborLists() * 
//            nxp * nyp * nfp *
//            getNumDataPatches());
//
//      //Update weights with momentum
//      for(int arborId=0;arborId<numberOfAxonalArborLists();arborId++){
//         status = updateWeights(arborId);  // Apply changes in weights
//         if (status==PV_BREAK) { break; }
//         assert(status==PV_SUCCESS);
//      }
//
//      //Clear dw after weights are updated
//      clear_dW();
//      //Reset numKernelActivations
//      for(int arbor_ID = 0; arbor_ID < numberOfAxonalArborLists(); arbor_ID++){
//         int numKernelIndices = getNumDataPatches();
//         int patchSize = nxp * nyp * nfp;
//#ifdef PV_USE_OPENMP_THREADS
//#pragma omp parallel for
//#endif
//         for(int ki = 0; ki < numKernelIndices; ki++){
//            for(int pi = 0; pi < patchSize; pi++){
//               numKernelActivations[arbor_ID][ki][pi] = 0;
//            }
//         }
//      }
//      //Reset batchIdx
//      batchIdx = 0;
//   }
//   else{
//      batchIdx ++;
//   }
//
//   // normalizeWeights(); // normalizeWeights call moved to HyPerCol::advanceTime loop, to allow for normalization of a group of connections
//
//   update_timer->stop();
//   return status;
//}

int MomentumConn::applyMomentum(int arbor_ID){
   int nExt = preSynapticLayer()->getNumExtended();
   const PVLayerLoc * loc = preSynapticLayer()->getLayerLoc();
   if(sharedWeights){
      int numKernels = getNumDataPatches();
      //Shared weights done in parallel, parallel in numkernels
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for(int kernelIdx = 0; kernelIdx < numKernels; kernelIdx++){
         pvwdata_t * dwdata_start  = get_dwDataHead(arbor_ID, kernelIdx);
         pvwdata_t * prev_dw_start = get_prev_dwDataHead(arbor_ID, kernelIdx);
         pvwdata_t * wdata_start   = get_wDataHead(arbor_ID, kernelIdx);
         if(!strcmp(momentumMethod, "simple")){
            for(int k = 0; k < nxp*nyp*nfp; k++){
               dwdata_start[k] += momentumTau * prev_dw_start[k] - momentumDecay*wdata_start[k];
            }
         }
         else if(!strcmp(momentumMethod, "viscosity")){
            for(int k = 0; k < nxp*nyp*nfp; k++){
               //dwdata_start[k] = momentumTau * (prev_dw_start[k] + dwdata_start[k]) * (1 - exp(-1.0/ momentumTau)) - momentumDecay*wdata_start[k];
               dwdata_start[k] = (prev_dw_start[k] * exp(-1.0/ momentumTau)) + dwdata_start[k] - momentumDecay*wdata_start[k];
            }
         }
         else if(!strcmp(momentumMethod, "alex")){
            for(int k = 0; k < nxp*nyp*nfp; k++){
               //weight_inc[i] := momW * weight_inc[i-1] - wc * epsW * weights[i-1] + epsW * weight_grads[i]
               //   weights[i] := weights[i-1] + weight_inc[i]
               dwdata_start[k] = momentumTau * prev_dw_start[k] - momentumDecay * getDWMax()* wdata_start[k] + dwdata_start[k];
            }
         }
      }
   }
   else{
      std::cout << "Warning: Momentum not implemented for non-shared weights, not implementing momentum\n";
   }
   return PV_SUCCESS;
}

//TODO checkpointing not working with batching, must write checkpoint exactly at period
int MomentumConn::checkpointWrite(const char * cpDir) {
   HyPerConn::checkpointWrite(cpDir);
   if (!plasticityFlag) return PV_SUCCESS;
   char filename[PV_PATH_MAX];
   int chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_prev_dW.pvp", cpDir, name);
   if(chars_needed >= PV_PATH_MAX) {
      if ( parent->icCommunicator()->commRank()==0 ) {
         fprintf(stderr, "HyPerConn::checkpointFilename error: path \"%s/%s_W.pvp\" is too long.\n", cpDir, name);
      }
      abort();
   }
   PVPatch *** patches_arg = sharedWeights ? NULL : get_wPatches();
   int status = writeWeights(patches_arg, prev_dwDataStart, getNumDataPatches(), filename, parent->simulationTime(), writeCompressedCheckpoints, /*last*/true);
   assert(status==PV_SUCCESS);
   return PV_SUCCESS;
}

int MomentumConn::checkpointRead(const char * cpDir, double * timeptr) {
   HyPerConn::checkpointRead(cpDir, timeptr);
   if (!plasticityFlag) return PV_SUCCESS;
   clearWeights(prev_dwDataStart, getNumDataPatches(), nxp, nyp, nfp);
   char * path = parent->pathInCheckpoint(cpDir, getName(), "_prev_dW.pvp");
   PVPatch *** patches_arg = sharedWeights ? NULL : get_wPatches();
   double filetime=0.0;
   int status = PV::readWeights(patches_arg, prev_dwDataStart, numberOfAxonalArborLists(), getNumDataPatches(), nxp, nyp, nfp, path, parent->icCommunicator(), &filetime, pre->getLayerLoc());
   if (parent->columnId()==0 && timeptr && *timeptr != filetime) {
      fprintf(stderr, "Warning: \"%s\" checkpoint has timestamp %g instead of the expected value %g.\n", path, filetime, *timeptr);
   }
   free(path);
   return status;
}

BaseObject * createMomentumConn(char const * name, HyPerCol * hc) {
   if (hc==NULL) { return NULL; }
   InitWeights * weightInitializer = getWeightInitializer(name, hc);
   NormalizeBase * weightNormalizer = getWeightNormalizer(name, hc);
   return new MomentumConn(name, hc, weightInitializer, weightNormalizer);
}

} // end namespace PV
