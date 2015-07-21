/*
 * SparsityLayerProbe.cpp
 *
 *  Created on: Apr 2, 2014
 *      Author: slundquist
 */

#include "SparsityLayerProbe.hpp"
#include "../layers/HyPerLayer.hpp"

namespace PV {

/**
 * @filename
 */
SparsityLayerProbe::SparsityLayerProbe(const char * probeName, HyPerCol * hc)
{
   initSparsityLayerProbe_base();
   LayerProbe::initialize(probeName, hc);
}

SparsityLayerProbe::SparsityLayerProbe()
   : LayerProbe()
{
   initSparsityLayerProbe_base();
   // Derived classes should call initStatsProbe
}

SparsityLayerProbe::~SparsityLayerProbe()
{
   free(sparsityVals);
   free(timeVals);
}


int SparsityLayerProbe::initSparsityLayerProbe_base() {
   sparsityVals = NULL;
   timeVals = NULL;
   bufIndex = -1; //-1 initialization since we're incrementing first before we put in data
   bufSize = 0;
   windowSize = 1000; //Default value of 1000, what should it be?
   calcNNZ = true;
   initSparsityVal = .01;
   ANNTargetLayer = NULL;
   return PV_SUCCESS;
}

int SparsityLayerProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = LayerProbe::ioParamsFillGroup(ioFlag);
   ioParam_windowSize(ioFlag);
   ioParam_calcNNZ(ioFlag);
   ioParam_initSparsityVal(ioFlag);
   return status;
}

void SparsityLayerProbe::ioParam_windowSize(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "windowSize", &windowSize, windowSize);
}

void SparsityLayerProbe::ioParam_calcNNZ(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "calcNNZ", &calcNNZ, calcNNZ);
}

void SparsityLayerProbe::ioParam_initSparsityVal(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "initSparsityVal", &initSparsityVal, initSparsityVal);
}

int SparsityLayerProbe::communicateInitInfo() {
   int status = LayerProbe::communicateInitInfo();
   //
   //Need to calculate the buffer size, which is based off the deltatime of the triggerLayer, if it exists
   if(triggerFlag){
      assert(triggerLayer);
      deltaUpdateTime = triggerLayer->getDeltaUpdateTime();
      //Never update
      if(deltaUpdateTime == -1){
         bufSize = 1;
      }
      else{
         double dBufSize = (float)windowSize/deltaUpdateTime;
         if(dBufSize < 1){
            fprintf(stderr, "SparsityLayerProbe %s: window size must be bigger than the trigger layer's delta update time (%f)", name, deltaUpdateTime);
            exit(EXIT_FAILURE);
         }
         bufSize = ceil(dBufSize);
      }
   }
   else{
      bufSize = ceil((float)windowSize/parent->getDeltaTime());
      deltaUpdateTime = parent->getDeltaTime();
   }
   //Allocate buffers
   sparsityVals = (float*) calloc(bufSize, sizeof(float));
   timeVals = (double*) calloc(bufSize, sizeof(double));
   //Initialize sparsityVals
   for(int i = 0; i < bufSize; i++){
      sparsityVals[i] = initSparsityVal;
   }

   //Check if attached layer is LCA for stats output
   ANNTargetLayer = dynamic_cast<ANNLayer*>(getTargetLayer());
   
   return status;
}

/**
 * 2 buffers (sparsityVals and timeVals) are circular buffers
 */
void SparsityLayerProbe::updateBufIndex(){
   bufIndex++;
   if(bufIndex == bufSize){
      bufIndex = 0;
   }
}

/**
 * @time
 */
int SparsityLayerProbe::outputState(double timed)
{
   int rank = 0;
   //Grab needed info
#ifdef PV_USE_MPI
   InterColComm * icComm = getTargetLayer()->getParent()->icCommunicator();
   MPI_Comm comm = icComm->communicator();
   rank = icComm->commRank();
   const int rcvProc = 0;
#endif // PV_USE_MPI
   const pvdata_t * buf = getTargetLayer()->getLayerData();
   int nk = getTargetLayer()->getNumNeurons();
   const PVLayerLoc * loc = getTargetLayer()->getLayerLoc();
   int numTotNeurons = loc->nxGlobal * loc->nyGlobal * loc->nf;
   //Update index
   updateBufIndex();
   //Calculating nnz method
   if (calcNNZ){
      int nnz = 0;
      for( int k=0; k<nk; k++ ) {
         int kex = kIndexExtended(k, loc->nx, loc->ny, loc->nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
         pvdata_t a = buf[kex];
         if(a > 0){
            nnz++;
         }
      }
#ifdef PV_USE_MPI
      //Sum all nnz across processors
      MPI_Allreduce(MPI_IN_PLACE, &nnz, 1, MPI_INT, MPI_SUM, comm);
#endif // PV_USE_MPI
      sparsityVals[bufIndex] = (float)nnz/numTotNeurons;
   }
   //Calculating mean of values method
   else{
      float sumVal = 0;
      for( int k=0; k<nk; k++ ) {
         int kex = kIndexExtended(k, loc->nx, loc->ny, loc->nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
         pvdata_t a = buf[kex];
         sumVal += a;
      }
#ifdef PV_USE_MPI
      //Sum all nnz across processors
      MPI_Allreduce(MPI_IN_PLACE, &sumVal, 1, MPI_FLOAT, MPI_SUM, comm);
#endif // PV_USE_MPI
      sparsityVals[bufIndex] = sumVal/numTotNeurons;
   }
   //Save timestep
   timeVals[bufIndex] = timed;
   //Write out to file probe
   if(rank == 0){
      fprintf(outputstream->fp, "%st==%6.1f Sparsity==%f", getMessage(), timed, getSparsity());
      if(ANNTargetLayer){
         fprintf(outputstream->fp, " VThresh==%f", ANNTargetLayer->getVThresh());
      }
      fprintf(outputstream->fp, "\n");
      fflush(outputstream->fp);
   }


   //Print out for testing
   //for(int i = 0; i < bufSize; i++){
   //   std::cout << timeVals[i] << ":" << sparsityVals[i] << "  ";
   //}
   //std::cout << "\n";
   return PV_SUCCESS;
}

//Getter functions for probe statstics
float SparsityLayerProbe::getSparsity(){
   //Find mean of entire buffer
   float sum = 0;
   for(int i = 0; i < bufSize; i++){
      sum += sparsityVals[i];
   }
   return sum/bufSize;
}

double SparsityLayerProbe::getUpdateTime(){
   return timeVals[bufIndex] + parent->getDeltaTime();
}

} // namespace PV
