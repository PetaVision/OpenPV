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
   momentumMethod = NULL;
   momentumDecay = 0;
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
   ioParam_momentumMethod(ioFlag);
   ioParam_momentumDecay(ioFlag);
   return status;
}

void MomentumConn::ioParam_momentumTau(enum ParamsIOFlag ioFlag){
   if(plasticityFlag){
      parent->ioParamValue(ioFlag, name, "momentumTau", &momentumTau, momentumTau);
   }
}

/**
 * @brief momentumMethod: The momentum method to use
 * @details Assuming a = dwMax * pre * post
 * simple: deltaW(t) = a + momentumTau * deltaW(t-1)
 * viscosity: deltaW(t) = momentumTau * (deltaW(t-1) + a) * (1-e^(-deltaT/momentumTau))
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
   
   //Update prev_dwData buffer
   assert(prev_dwDataStart);
   //After reduce, copy over to prev_dwData
   std::memcpy(*prev_dwDataStart, *get_dwDataStart(),
         sizeof(pvwdata_t) *
         numberOfAxonalArborLists() * 
         nxp * nyp * nfp *
         getNumDataPatches());

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
         pvwdata_t * dwdata_start  = get_dwDataHead(arbor_ID, kernelIdx);
         pvwdata_t * prev_dw_start = get_prev_dwDataHead(arbor_ID, kernelIdx);
         pvwdata_t * wdata_start   = get_wData(arbor_ID, kernelIdx);
         if(!strcmp(momentumMethod, "simple")){
            for(int k = 0; k < nxp*nyp*nfp; k++){
               dwdata_start[k] += momentumTau * prev_dw_start[k] - momentumDecay*wdata_start[k];
            }
         }
         else if(!strcmp(momentumMethod, "viscosity")){
            for(int k = 0; k < nxp*nyp*nfp; k++){
               dwdata_start[k] = momentumTau * (prev_dw_start[k] + dwdata_start[k]) * (1 - exp(- parent->getDeltaTime() / momentumTau)) - momentumDecay*wdata_start[k];
            }
         }
         else if(!strcmp(momentumMethod, "alex")){
            for(int k = 0; k < nxp*nyp*nfp; k++){
               dwdata_start[k] = momentumTau * prev_dw_start[k] - (1-momentumDecay) * getDWMax()* wdata_start[k] - dwdata_start[k];
            }
         }
      }
   }
   else{
      std::cout << "Momentum not implemented for non-shared weights\n";
      exit(-1);
   }
   return PV_SUCCESS;
}



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

} // end namespace PV
