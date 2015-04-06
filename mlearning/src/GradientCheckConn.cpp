/*
 * GradientCheckProbe.cpp
 * This probe estimates the gradient numerically and checks it with the connection's dw weights.
 * Numerical solution equation:
 * E = (C(\Theta + \epsilon) - C(\Theta)) / \epsilon
 * where \Theta is the vector of weights and epsilon is a small value
 *
 * Author: slundquist
 */


#include "GradientCheckConn.hpp"

namespace PVMLearning{

GradientCheckConn::GradientCheckConn() {
    initialize_base();
}

GradientCheckConn::GradientCheckConn(const char * name, PV::HyPerCol * hc) {
   initialize_base();
   initialize(name, hc, NULL, NULL);
}

int GradientCheckConn::initialize_base() {
   firstRun = true;
   secondRun = true;
   estLayerName = NULL;
   gtLayerName = NULL;
   estLayer = NULL;
   gtLayer = NULL;
   epsilon = 1e-4;
   prevIdx = -1;
   prevWeightVal = 0;
   return PV_SUCCESS;
}

int GradientCheckConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = PV::HyPerConn::ioParamsFillGroup(ioFlag);
   ioParam_estLayerName(ioFlag);
   ioParam_gtLayerName(ioFlag);
   return status;
}

int GradientCheckConn::communicateInitInfo() {
   int status = PV::HyPerConn::communicateInitInfo();

   PV::Communicator * comm = parent->icCommunicator();
   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();
   const int nProcs = nxProcs * nyProcs;
   if(nProcs != 1){
      std::cout << "Error, GradientCheckConn cannot be ran with MPI\n";
   }


   estLayer = parent->getLayerFromName(estLayerName);
   if (estLayer ==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: estLayerName \"%s\" is not a layer in the HyPerCol.\n",
                 parent->parameters()->groupKeywordFromName(name), name, estLayerName);
      }
#ifdef PV_USE_MPI
      MPI_Barrier(parent->icCommunicator()->communicator());
#endif
      exit(EXIT_FAILURE);
   }

   gtLayer = parent->getLayerFromName(gtLayerName);
   if (gtLayer ==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: gtLayerName \"%s\" is not a layer in the HyPerCol.\n",
                 parent->parameters()->groupKeywordFromName(name), name, gtLayerName);
      }
#ifdef PV_USE_MPI
      MPI_Barrier(parent->icCommunicator()->communicator());
#endif
      exit(EXIT_FAILURE);
   }

   assert(gtLayer->getNumNeurons() == estLayer->getNumNeurons());

   return status;
}

int GradientCheckConn::allocateDataStructures() {
   int status = PV::HyPerConn::allocateDataStructures();

   //Check num weights against num timesteps
   int numDataPatches = getNumDataPatches();
   int numWeights = nxp * nyp * nfp;
   int numArbors = numAxonalArborLists;
 
   if(parent->getFinalStep() - parent->getInitialStep() > numDataPatches * numWeights * numArbors + 2){
      std::cout << "Maximum number of steps for GradientCheckConn is " << numDataPatches * numWeights * numArbors + 2 << "\n";
      exit(-1);
   }
}

void GradientCheckConn::ioParam_estLayerName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "estLayerName", &estLayerName);
}

void GradientCheckConn::ioParam_gtLayerName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "gtLayerName", &gtLayerName);
}

void GradientCheckConn::ioParam_plasticityFlag(enum ParamsIOFlag ioFlag){
   plasticityFlag = true;
   parent->parameters()->handleUnnecessaryParameter(name, "plasticityFlag", plasticityFlag);
}

//Connections update first
int GradientCheckConn::updateState(double time, double dt){
   int status = PV_SUCCESS;
   int weightIdx = parent->getCurrentStep() - parent->getInitialStep() - 2;
   std::cout << "weightIdx " << weightIdx << "\n";
   int numPatch = nxp * nyp * nfp;
   int numData = getNumDataPatches();
   int arborIdx = weightIdx / (numPatch * numData);
   int dataIdx = (weightIdx / numPatch) % numData;
   int patchIdx = weightIdx % numPatch;

   if(firstRun){
      clear_dW();
      firstRun = false;
      return PV_SUCCESS;
   }
   
   //Grab cost from previous timestep
   if(secondRun){
      //First run does regular updateState to calculate dw buffer
      for(int arborId=0;arborId<numberOfAxonalArborLists();arborId++) {
         status = calc_dW(arborId);        // Calculate changes in weights
         if (status==PV_BREAK) { break; }
         assert(status == PV_SUCCESS);
      }
      for (int arborID = 0; arborID < numberOfAxonalArborLists(); arborID++) {
         status = reduceKernels(arborID); // combine partial changes in each column
         if (status == PV_BREAK) {
            break;
         }
         assert(status == PV_SUCCESS);
      }
      //No update weights
      origCost = getCost();
      secondRun = false;
   }

   //Does not update after first run
   //Check if we are in bounds for non-shared weights
   if(!sharedWeights){
      PVPatch* weights = getWeights(dataIdx, arborIdx);
      //Calculate x and y of patchIdx and compare it to offset
      int xPatchIdx = kxPos(patchIdx, nxp, nyp, nfp);
      int yPatchIdx = kyPos(patchIdx, nxp, nyp, nfp);
      int xOffsetIdx = kxPos(weights->offset, nxp, nyp, nfp);
      int yOffsetIdx = kyPos(weights->offset, nxp, nyp, nfp);

      //If index is oob, skip
      if(xPatchIdx < xOffsetIdx || xPatchIdx >= xOffsetIdx + weights->nx ||
         yPatchIdx < yOffsetIdx || yPatchIdx >= yOffsetIdx + weights->ny){
         return PV_SUCCESS;
      }
   }

   //Calculate difference in numerical method and backprop method
   if(prevIdx != -1){
      currCost = getCost();
      //Check for accuracy
      float numGradient = (currCost - origCost)/epsilon;
      float backpropGradient = get_dwDataStart()[0][prevIdx] / dWMax;
      if(numGradient + backpropGradient >= .1){
         std::cout << "Numerical gradient: " << numGradient << "  Backprop gradient: " << backpropGradient << "\n";
         exit(-1);

      }
   }

   //Restore weight
   if(prevIdx != -1){
      get_wDataStart()[0][prevIdx] = prevWeightVal;
   }

   //Set next weight if not the end
   if(weightIdx < numberOfAxonalArborLists() * numData * numPatch){
      prevWeightVal = get_wDataStart()[0][weightIdx];
      prevIdx = weightIdx;
      get_wDataStart()[0][weightIdx] += epsilon;
   }
   else{
      std::cout << "END\n";
   }

   return status;
}

float GradientCheckConn::getCost(){
   //TODO condional checking cost function
   return getSqErrCost();
}

float GradientCheckConn::getSqErrCost(){
   float* estA = estLayer->getActivity();
   float* gtA = gtLayer->getActivity();
   const PVLayerLoc * gtLoc = gtLayer->getLayerLoc();
   const PVLayerLoc * estLoc = estLayer->getLayerLoc();
   float sumsq = 0;

   for(int kRes = 0; kRes < estLayer->getNumNeurons(); kRes++){
      int estExt = kIndexExtended(kRes, estLoc->nx, estLoc->ny, estLoc->nf, estLoc->halo.lt, estLoc->halo.rt, estLoc->halo.dn, estLoc->halo.up);
      int gtExt = kIndexExtended(kRes, gtLoc->nx, gtLoc->ny, gtLoc->nf, gtLoc->halo.lt, gtLoc->halo.rt, gtLoc->halo.dn, gtLoc->halo.up);
      sumsq += pow(gtA[gtExt] - estA[estExt], 2); 
   }

//#ifdef PV_USE_MPI
//   MPI_Allreduce(MPI_IN_PLACE, sumsq, 1, MPI_FLOAT, MPI_SUM, parent->icCommunicator()->communicator());
//#endif

   return .5 * sumsq;
}






}  // end of namespace PV block
