/*
 * MomentumConn.cpp
 *
 *  Created on: Feburary 27, 2014
 *      Author: slundquist
 */

#include "MomentumConn.hpp"
#include <cstring>

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
}

int MomentumConn::initialize_base() {
   prev_dwDataStart = NULL;
   momentumTau = .25;
   momentumPeriod = 1;
   momentumPeriodIdx = 0;
   momentumMethod = NULL;
   return PV_SUCCESS;
}

int MomentumConn::allocateDataStructures(){
   HyPerConn::allocateDataStructures();
   if (!plasticityFlag) return PV_SUCCESS;
   int sx = nfp;
   int sy = sx * nxp;
   int sp = sy * nyp;
   int nPatches = getNumDataPatches();

   const int numAxons = numberOfAxonalArborLists();

   //Allocate dw buffer for previous dw
   prev_dwDataStart = (pvwdata_t **) calloc(numAxons, sizeof(pvwdata_t *));
   if( prev_dwDataStart == NULL ) {
      createArborsOutOfMemory();
      assert(false);
   }
   prev_dwDataStart[0] = (pvwdata_t*) calloc(numAxons * nxp * nyp * nfp * nPatches, sizeof(pvwdata_t));
   assert(prev_dwDataStart[0] != NULL);
   for (int arborId = 0; arborId < numAxons; arborId++) {
      prev_dwDataStart[arborId] = (prev_dwDataStart[0] + sp * nPatches * arborId);
      assert(prev_dwDataStart[arborId] != NULL);
   } // loop over arbors

   return PV_SUCCESS;
}

int MomentumConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   ioParam_momentumTau(ioFlag);
   ioParam_momentumPeriod(ioFlag);
   ioParam_momentumMethod(ioFlag);
   return status;
}

void MomentumConn::ioParam_momentumTau(enum ParamsIOFlag ioFlag){
   if(plasticityFlag){
      parent->ioParamValue(ioFlag, name, "momentumTau", &momentumTau, momentumTau);
   }
}

/**
 * @brief momentumPeriod: The number of weight updates before the momentum term updates
 */
void MomentumConn::ioParam_momentumPeriod(enum ParamsIOFlag ioFlag){
   if(plasticityFlag){
      parent->ioParamValue(ioFlag, name, "momentumPeriod", &momentumPeriod, momentumPeriod);
      if(momentumPeriod < 1){
         momentumPeriod = 1;
      }
   }
}

/**
 * @brief momentumMethod: The momentum method to use
 * @details Assuming a = dwMax * pre * post
 * simple: deltaW(t) = a + momentumTau * deltaW(t-1)
 * viscosity: deltaW(t) = momentumTau * (deltaW(t-1) + a) * (1-e^(-deltaT/momentumTau))
 */
void MomentumConn::ioParam_momentumMethod(enum ParamsIOFlag ioFlag){
   if(plasticityFlag){
      parent->ioParamStringRequired(ioFlag, name, "momentumMethod", &momentumMethod);
      if(strcmp(momentumMethod, "simple") != 0 && strcmp(momentumMethod, "viscosity") != 0){
         std::cout << "MomentumConn " << name << ": momentumMethod of " << momentumMethod << " is not known, options are \"simple\" and \"viscosity\"\n";
         exit(-1);
      }
   }
}


//Copied from HyPerConn, only change is a memcpy from dwweights to prev_dwweights
int MomentumConn::updateState(double time, double dt){
   int status = PV_SUCCESS;
   if( !plasticityFlag ) {
      return status;
   }
   update_timer->start();

   if (!combine_dW_with_W_flag) { clear_dW(); }
   for(int arborId=0;arborId<numberOfAxonalArborLists();arborId++) {
      status = calc_dW(arborId);        // Calculate changes in weights
      if (status==PV_BREAK) { break; }
      assert(status == PV_SUCCESS);
   }

   bool needSynchronizing = keepKernelsSynchronized_flag;
   needSynchronizing |= sharedWeights && (parent->simulationTime() >= parent->getStopTime()-parent->getDeltaTime());
   if (needSynchronizing) {
      for (int arborID = 0; arborID < numberOfAxonalArborLists(); arborID++) {
         status = reduceKernels(arborID); // combine partial changes in each column
         if (status == PV_BREAK) {
            break;
         }
         assert(status == PV_SUCCESS);
      }
   }

   //Apply momentum
   for (int arborID = 0; arborID < numberOfAxonalArborLists(); arborID++) {
      status = applyMomentum(arborID);
      if (status == PV_BREAK) {
         break;
      }
      assert(status == PV_SUCCESS);
   }
   
   //TODO: when updating prev_dwDataStart, make sure to get average
   //Update prev_dwData buffer
   if(momentumPeriodIdx >= momentumPeriod - 1){
      momentumPeriodIdx = 0;
      assert(prev_dwDataStart);
      //After reduce, copy over to prev_dwData
      std::memcpy(*prev_dwDataStart, *get_dwDataStart(),
            sizeof(pvwdata_t) *
            numberOfAxonalArborLists() * 
            nxp * nyp * nfp *
            getNumDataPatches());
   }
   else{
      momentumPeriodIdx++;
   }

   for(int arborId=0;arborId<numberOfAxonalArborLists();arborId++){
      status = updateWeights(arborId);  // Apply changes in weights
      if (status==PV_BREAK) { break; }
      assert(status==PV_SUCCESS);
   }
   // normalizeWeights(); // normalizeWeights call moved to HyPerCol::advanceTime loop, to allow for normalization of a group of connections

   update_timer->stop();
   return status;
}

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
         pvwdata_t * dwdata_start = get_dwDataHead(arbor_ID, kernelIdx);
         pvwdata_t* prev_dw_start = get_prev_dwDataHead(arbor_ID, kernelIdx);
         if(!strcmp(momentumMethod, "simple")){
            for(int k = 0; k < nxp*nyp*nfp; k++){
               dwdata_start[k] += momentumTau * prev_dw_start[k];
            }
         }
         else if(!strcmp(momentumMethod, "viscosity")){
            for(int k = 0; k < nxp*nyp*nfp; k++){
               dwdata_start[k] = momentumTau * (prev_dw_start[k] + dwdata_start[k]) * (1 - exp(- parent->getDeltaTime() / momentumTau));
            }
         }
      }
   }
   else{
//      //No clobbering for non-shared weights
//#ifdef PV_USE_OPENMP_THREADS
//#pragma omp parallel for
//#endif
//      for(int kExt=0; kExt<nExt;kExt++) {
//         applyIndMomentum(arbor_ID, kExt);
//      }
      std::cout << "Momentum not implemented for non-shared weights\n";
      exit(-1);
   }
   return PV_SUCCESS;
}


//int MomentumConn::applyIndMomentum(int arbor_ID, int kExt){
//   int sya = (post->getLayerLoc()->nf * (post->getLayerLoc()->nx + post->getLayerLoc()->halo.lt + post->getLayerLoc()->halo.rt));
//   PVPatch * weights = getWeights(kExt,arbor_ID);
//
//   //Offset, since post is in res space, should be right for both mask and post layer
//   size_t offset = getAPostOffset(kExt, arbor_ID);
//
//   int ny = weights->ny;
//   int nk = weights->nx * nfp;
//
//   int kernelIndex = patchIndexToDataIndex(kExt);
//   pvwdata_t * dwdata = get_dwData(arbor_ID, kExt);
//   pvwdata_t * prev_dwdata = get_prev_dwData(arbor_ID, kExt);
//   int lineoffsetw = 0;
//   for( int y=0; y<ny; y++ ) {
//      for( int k=0; k<nk; k++ ) {
//         const pvwdata_t prev_dw = prev_dwdata[lineoffsetw + k];
//         if(!strcmp(momentumMethod, "simple")){
//            dwdata[lineoffsetw + k] += momentumTau * prev_dw;
//         }
//         else if(!strcmp(momentumMethod, "viscosity")){
//            dwdata[lineoffsetw + k] = momentumTau * (prev_dw + dwdata[lineoffsetw + k]) * (1 - exp(- parent->getDeltaTime() / momentumTau));
//         }
//      }
//      lineoffsetw += syp;
//   }
//   return PV_SUCCESS;
//}

//TODO Checkpointing

} // end namespace PV
