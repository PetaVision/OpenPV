/*
 * BatchConn.cpp
 *
 *  Created on: Nov 17, 2010
 *      Author: pschultz
 */

#include "BatchConn.hpp"
#include <connections/PlasticCloneConn.hpp>

namespace PV {

BatchConn::BatchConn() {
    initialize_base();
}

BatchConn::BatchConn(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc, NULL, NULL);
}

int BatchConn::initialize_base() {
   batchPeriod = 1;
   batchIdx = 0;
   return PV_SUCCESS;
}  // end of BatchConn::initialize_base()

int BatchConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   ioParam_batchPeriod(ioFlag);
   return status;
}

void BatchConn::ioParam_batchPeriod(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "plasticityFlag"));
   if(plasticityFlag){
      parent->ioParamValue(ioFlag, name, "batchPeriod", &batchPeriod, batchPeriod);
   }
}

//Reduce kerenls should never be getting called except for checkpoint write, which is being overwritten.
int BatchConn::reduceKernels(const int arborID){
   fprintf(stderr, "Error:  BatchConn calling reduceKernels\n");
   exit(1);
}

void BatchConn::sumKernelActivations(){
   HyPerConn::reduceNumKernelActivations();
}

//No normalize reduction here, just summing up
int BatchConn::sumKernels(const int arborID) {
   assert(sharedWeights && plasticityFlag);
   Communicator * comm = parent->icCommunicator();
   const MPI_Comm mpi_comm = comm->communicator();
   int ierr;
   const int nxProcs = comm->numCommColumns();
   const int nyProcs = comm->numCommRows();
   const int nProcs = nxProcs * nyProcs;
   if (nProcs == 1){
      normalize_dW(arborID);
      return PV_BREAK;
   }
   const int numPatches = getNumDataPatches();
   const size_t patchSize = nxp*nyp*nfp;
   const size_t localSize = numPatches * patchSize;
   const size_t arborSize = localSize * this->numberOfAxonalArborLists();

#ifdef PV_USE_MPI
   ierr = MPI_Allreduce(MPI_IN_PLACE, this->get_dwDataStart(arborID), arborSize, MPI_FLOAT, MPI_SUM, mpi_comm);
#endif
   return PV_BREAK;
}

//Copied from HyPerConn
int BatchConn::defaultUpdate_dW(int arbor_ID) {
   int nExt = preSynapticLayer()->getNumExtended();
   const PVLayerLoc * loc = preSynapticLayer()->getLayerLoc();

   //Calculate x and y cell size
   int xCellSize = zUnitCellSize(pre->getXScale(), post->getXScale());
   int yCellSize = zUnitCellSize(pre->getYScale(), post->getYScale());
   int nxExt = loc->nx + loc->halo.lt + loc->halo.rt;
   int nyExt = loc->ny + loc->halo.up + loc->halo.dn;
   int nf = loc->nf;
   int numKernels = getNumDataPatches();

   //Shared weights done in parallel, parallel in numkernels
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
   for(int kernelIdx = 0; kernelIdx < numKernels; kernelIdx++){
      //Calculate xCellIdx, yCellIdx, and fCellIdx from kernelIndex
      int kxCellIdx = kxPos(kernelIdx, xCellSize, yCellSize, nf);
      int kyCellIdx = kyPos(kernelIdx, xCellSize, yCellSize, nf);
      int kfIdx = featureIndex(kernelIdx, xCellSize, yCellSize, nf);
      //Loop over all cells in pre ext
      int kyIdx = kyCellIdx;
      int yCellIdx = 0;
      while(kyIdx < nyExt){
         int kxIdx = kxCellIdx;
         int xCellIdx = 0;
         while(kxIdx < nxExt){
            //Calculate kExt from ky, kx, and kf
            int kExt = kIndex(kxIdx, kyIdx, kfIdx, nxExt, nyExt, nf);
            defaultUpdateInd_dW(arbor_ID, kExt);
            xCellIdx++;
            kxIdx = kxCellIdx + xCellIdx * xCellSize;
         }
         yCellIdx++;
         kyIdx = kyCellIdx + yCellIdx * yCellSize;
      }
   }

   return PV_SUCCESS;
}

int BatchConn::normalize_dW(int arbor_ID){
   if (sharedWeights) {
      for( int loop_arbor; loop_arbor < numberOfAxonalArborLists(); loop_arbor++){
         // Divide by numKernelActivations in this timestep
         int numKernelIndices = getNumDataPatches();
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for( int kernelindex=0; kernelindex<numKernelIndices; kernelindex++ ) {
            //Calculate pre feature index from patch index
            int numpatchitems = nxp*nyp*nfp;
            pvwdata_t * dwpatchdata = get_dwDataHead(loop_arbor,kernelindex);
            for( int n=0; n<numpatchitems; n++ ) {
               long divisor = numKernelActivations[loop_arbor][kernelindex][n];
               //Divisor should not overflow
               if(divisor != 0){
                  dwpatchdata[n] /= divisor;
               }
               else{
                  dwpatchdata[n] = 0;
               }
            }
         }
      }
   }
   return PV_BREAK;
}

//Copied from HyPerConn updateState
int BatchConn::updateState(double time, double dt){
   int status = PV_SUCCESS;
   if( !plasticityFlag ) {
      return status;
   }
   update_timer->start();

   //Calculate dw for all arbors, will accumulate into dw
   for(int arborId=0;arborId<numberOfAxonalArborLists();arborId++) {
      status = calc_dW(arborId);        // Calculate changes in weights
      if (status==PV_BREAK) { break; }
      assert(status == PV_SUCCESS);
   }

   //Only update weights if batchIdx reaches batchPeriod
   std::cout << "batchIdx: " << batchIdx << "\n";
   if (batchIdx >= batchPeriod-1){
      std::cout << "Updating Batch\n";
      //Do reduction when updating
      sumKernels(0); //Sum all kernel activations
      sumKernelActivations(); //Doing sum here as oppsed to average
      //Normalize based on kernel activations
      normalize_dW(0);
      for(int arborId=0;arborId<numberOfAxonalArborLists();arborId++){
         status = updateWeights(arborId);  // Apply changes in weights
         if (status==PV_BREAK) { break; }
         assert(status==PV_SUCCESS);
      }
      
      //Clear dw after weights are updated
      clear_dW();

      //Reset numKernelActivations
      for(int arbor_ID = 0; arbor_ID < numberOfAxonalArborLists(); arbor_ID++){
         int numKernelIndices = getNumDataPatches();
         int patchSize = nxp * nyp * nfp;
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for(int ki = 0; ki < numKernelIndices; ki++){
            for(int pi = 0; pi < patchSize; pi++){
               numKernelActivations[arbor_ID][ki][pi] = 0;
            }
         }
      }
      //Reset batchIdx
      batchIdx = 0;
   }
   else{
      batchIdx ++;
   }

   update_timer->stop();
   return status;
}

}  // end of namespace PV block
