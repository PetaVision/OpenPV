/*
 * MLPOutputLayer.cpp
 * Author: slundquist
 */

#include "MLPOutputLayer.hpp"
#include <iostream>

namespace PVMLearning {
MLPOutputLayer::MLPOutputLayer(){
   initialize_base();
}
MLPOutputLayer::MLPOutputLayer(const char * name, PV::HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

MLPOutputLayer::~MLPOutputLayer(){
   if(classBuffer){
      free(classBuffer);
   }
}

int MLPOutputLayer::initialize_base()
{
   localTarget = true;
   classBuffer = NULL;
   gtLayer = NULL;
   numRight = 0;
   numWrong = 0;
   progressNumRight = 0;
   progressNumWrong = 0;
   numTotPos = 0;
   numTotNeg = 0;
   truePos = 0;
   trueNeg = 0;
   progressNumTotPos = 0;
   progressNumTotNeg = 0;
   progressTruePos = 0;
   progressTrueNeg = 0;
   statProgressPeriod = 0; //Never print progress
   nextStatProgress = 0;
   return PV_SUCCESS;
}

int MLPOutputLayer::initialize(const char * name, PV::HyPerCol * hc)
{
   int status = MLPSigmoidLayer::initialize(name, hc);
   if(statProgressPeriod > 0){
      nextStatProgress = hc->getStartTime() + statProgressPeriod;
   }
   return status;
}

//TODO checkpoint stats and nextStatProgress

int MLPOutputLayer::communicateInitInfo(){
   int status = MLPSigmoidLayer::communicateInitInfo();
   if(statProgressPeriod > 0){
      gtLayer = parent->getLayerFromName(gtLayername);
      if (gtLayer==NULL) {
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: gtLayername \"%s\" is not a layer in the HyPerCol.\n",
                    getKeyword(), name, gtLayername);
         }
#ifdef PV_USE_MPI
         MPI_Barrier(parent->icCommunicator()->communicator());
#endif
         exit(EXIT_FAILURE);
      }

      const PVLayerLoc * srcLoc = gtLayer->getLayerLoc();
      const PVLayerLoc * loc = getLayerLoc();
      assert(srcLoc != NULL && loc != NULL);
      if (srcLoc->nxGlobal != loc->nxGlobal || srcLoc->nyGlobal != loc->nyGlobal || srcLoc->nf != loc->nf) {
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: gtLayerName \"%s\" does not have the same dimensions.\n",
                    getKeyword(), name, gtLayername);
            fprintf(stderr, "    original (nx=%d, ny=%d, nf=%d) versus (nx=%d, ny=%d, nf=%d)\n",
                    srcLoc->nxGlobal, srcLoc->nyGlobal, srcLoc->nf, loc->nxGlobal, loc->nyGlobal, loc->nf);
         }
#ifdef PV_USE_MPI
         MPI_Barrier(parent->icCommunicator()->communicator());
#endif
         exit(EXIT_FAILURE);
      }
   }
   return(status);
}

int MLPOutputLayer::allocateDataStructures() {
   int status = MLPSigmoidLayer::allocateDataStructures();
   //Allocate buffer size of this layer's nf (number of output classes)
   int nf = getLayerLoc()->nf;
   if(!localTarget){
      classBuffer = (pvdata_t*) malloc(nf * sizeof(pvdata_t));
   }
   //Grab gtLayername if it exists

   return status;
}

int MLPOutputLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = MLPSigmoidLayer::ioParamsFillGroup(ioFlag);
   ioParam_LocalTarget(ioFlag);
   ioParam_StatProgressPeriod(ioFlag);
   return status;
}

void MLPOutputLayer::ioParam_LocalTarget(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "localTarget", &localTarget, localTarget);
}

void MLPOutputLayer::ioParam_StatProgressPeriod(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "statProgressPeriod", &statProgressPeriod, statProgressPeriod);
   if(statProgressPeriod > 0){
      ioParam_GTLayername(ioFlag);
   }
}

void MLPOutputLayer::ioParam_GTLayername(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "gtLayername", &gtLayername);
}

void MLPOutputLayer::binaryNonlocalStats(){
   const PVLayerLoc * loc = getLayerLoc();
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   assert(nf == 1);
   int numNeurons = getNumNeurons();
   pvdata_t * A = getCLayer()->activity->data;
   pvdata_t * gtA = gtLayer->getCLayer()->activity->data;
   float sumsq = 0;
   float sum = 0;
   float gtSum = 0;
   int currNumRight = 0;
   int currNumWrong = 0;
   int totNum = 0;

   //Only go through restricted
   //Calculate the sum squared error
   for(int ni = 0; ni < numNeurons; ni++){
      int nExt = kIndexExtended(ni, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
      int fi = featureIndex(nExt, nx+loc->halo.lt+loc->halo.rt, ny+loc->halo.dn+loc->halo.up, nf);
      //Sum over x and y direction
      sumsq += pow(A[nExt] - gtA[nExt], 2);
      //Sum over activity to find mean
      sum += A[nExt];
      gtSum += gtA[nExt];
   }

#ifdef PV_USE_MPI
   MPI_Allreduce(MPI_IN_PLACE, &sumsq, 1, MPI_FLOAT, MPI_SUM, parent->icCommunicator()->communicator());
   MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_FLOAT, MPI_SUM, parent->icCommunicator()->communicator());
   MPI_Allreduce(MPI_IN_PLACE, &gtSum, 1, MPI_FLOAT, MPI_SUM, parent->icCommunicator()->communicator());
#endif // PV_USE_MPI
   //Normalize sum to find mean
   sum /= loc->nxGlobal * loc->nyGlobal;
   gtSum /= loc->nxGlobal * loc->nyGlobal;
   //gtSum should be the same as the values
   assert(gtSum == gtA[0]);

   //Calculate stats
   if(sum < 0 && gtSum < 0){
      currNumRight++;
   }
   else if(sum > 0 && gtSum > 0){
      currNumRight++;
   }
   else{
      currNumWrong++;
   }
#ifdef PV_USE_MPI
   MPI_Allreduce(MPI_IN_PLACE, &currNumRight, 1, MPI_INT, MPI_SUM, parent->icCommunicator()->communicator());
   MPI_Allreduce(MPI_IN_PLACE, &currNumWrong, 1, MPI_INT, MPI_SUM, parent->icCommunicator()->communicator());
#endif // PV_USE_MPI
   numRight += currNumRight;
   numWrong += currNumWrong;
   progressNumRight += currNumRight;
   progressNumWrong += currNumWrong;
   //Print if need
   float timef = parent->simulationTime();
   if(timef >= nextStatProgress){
      //Update nextStatProgress
      nextStatProgress += statProgressPeriod;
      if (parent->columnId()==0) {
         float totalScore = 100*float(numRight)/float(numRight+numWrong);
         float progressScore = 100*float(progressNumRight)/float(progressNumRight+progressNumWrong);
         fprintf(stdout, "time:%f  layer:\"%s\"  total:%f%%  progressStep:%f%%  energy:%f\n", timef, name, totalScore, progressScore, sumsq/2);
      }
      //Reset progressStats
      progressNumRight = 0;
      progressNumWrong = 0;
   }
}

void MLPOutputLayer::multiclassNonlocalStats(){
   const PVLayerLoc * loc = getLayerLoc();
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int numNeurons = getNumNeurons();
   pvdata_t * A = getCLayer()->activity->data;
   pvdata_t * gtA = gtLayer->getCLayer()->activity->data;
   float sumsq = 0;
   //Winner take all in the output layer
   int currNumRight = 0;
   int currNumWrong = 0;
   assert(classBuffer);
   //Clear classBuffer
   for(int i = 0; i < nf; i++){
      classBuffer[i] = 0;
   }
   //Only go through restricted
   //Calculate the sum squared error
   for(int ni = 0; ni < numNeurons; ni++){
      int nExt = kIndexExtended(ni, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
      int fi = featureIndex(nExt, nx+loc->halo.lt+loc->halo.rt, ny+loc->halo.dn+loc->halo.up, nf);
      //Sum over x and y direction
      classBuffer[fi] += A[nExt];
      sumsq += pow(A[nExt] - gtA[nExt], 2);
   }
   //Normalize classBuffer to find mean
   for(int i = 0; i < nf; i++){
      classBuffer[i] /= nx*ny;
   }
   //Reduce all classBuffers through a mean
#ifdef PV_USE_MPI
   MPI_Allreduce(MPI_IN_PLACE, &sumsq, 1, MPI_FLOAT, MPI_SUM, parent->icCommunicator()->communicator());
   MPI_Allreduce(MPI_IN_PLACE, classBuffer, nf, MPI_FLOAT, MPI_SUM, parent->icCommunicator()->communicator());
   //Normalize classBuffer across processors
   for(int i = 0; i < nf; i++){
      classBuffer[i] /= parent->icCommunicator()->commSize();
   }
#endif // PV_USE_MPI
   //Find max
   float estMaxF = -1000;
   int estMaxFi = -1;
   float actualMaxF = -1000;
   int actualMaxFi = -1;
   for(int i = 0; i < nf; i++){
      if(classBuffer[i] >= estMaxF){
         estMaxF = classBuffer[i];
         estMaxFi = i;
      }
      int nExt = kIndex(loc->halo.lt, loc->halo.up, i, nx+loc->halo.lt+loc->halo.rt, ny+loc->halo.dn+loc->halo.up, nf);
      if(gtA[nExt] >= actualMaxF){
         actualMaxF = gtA[nExt];
         actualMaxFi = i;
      }
   }
   //Calculate stats
   //Found winning feature, compare to ground truth
   if(estMaxFi == actualMaxFi){
      currNumRight++;
   }
   else{
      currNumWrong++;
   }
#ifdef PV_USE_MPI
   MPI_Allreduce(MPI_IN_PLACE, &currNumRight, 1, MPI_INT, MPI_SUM, parent->icCommunicator()->communicator());
   MPI_Allreduce(MPI_IN_PLACE, &currNumWrong, 1, MPI_INT, MPI_SUM, parent->icCommunicator()->communicator());
#endif // PV_USE_MPI
   numRight += currNumRight;
   numWrong += currNumWrong;
   progressNumRight += currNumRight;
   progressNumWrong += currNumWrong;
   //Print if need
   float timef = parent->simulationTime();
   if(timef >= nextStatProgress){
      //Update nextStatProgress
      nextStatProgress += statProgressPeriod;
      if (parent->columnId()==0) {
         float totalScore = 100*float(numRight)/float(numRight+numWrong);
         float progressScore = 100*float(progressNumRight)/float(progressNumRight+progressNumWrong);
         fprintf(stdout, "time:%f  layer:\"%s\"  total:%f%%  progressStep:%f%%  energy:%f\n", timef, name, totalScore, progressScore, sumsq/2);
      }
      //Reset progressStats
      progressNumRight = 0;
      progressNumWrong = 0;
   }
}

void MLPOutputLayer::binaryLocalStats(){
   const PVLayerLoc * loc = getLayerLoc();
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int numNeurons = getNumNeurons();
   pvdata_t * A = getCLayer()->activity->data;
   pvdata_t * gtA = gtLayer->getCLayer()->activity->data;
   float sumsq = 0;

   assert(nf == 1);
   int currNumTotPos = 0;
   int currNumTotNeg = 0;
   int currTruePos = 0;
   int currTrueNeg = 0;
   for(int ni = 0; ni < numNeurons; ni++){
      int nExt = kIndexExtended(ni, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
      //DCR
      if(gtA[nExt] == 0){
         continue;
         //Note that sumsq doesn't get updated in this case, so a dcr doesn't contribute to the score at all
      }
      //Negative
      else if(gtA[nExt] == -1){
         currNumTotNeg++;
         if(A[nExt] < 0){
            currTrueNeg++;
         }
      }
      //Positive
      else if(gtA[nExt] == 1){
         currNumTotPos++;
         if(A[nExt] > 0){
            currTruePos++;
         }
      }
      sumsq += pow(A[nExt] - gtA[nExt], 2);
   }
   //Do MPI
#ifdef PV_USE_MPI
   MPI_Allreduce(MPI_IN_PLACE, &currNumTotPos, 1, MPI_INT, MPI_SUM, parent->icCommunicator()->communicator());
   MPI_Allreduce(MPI_IN_PLACE, &currNumTotNeg, 1, MPI_INT, MPI_SUM, parent->icCommunicator()->communicator());
   MPI_Allreduce(MPI_IN_PLACE, &currTruePos, 1, MPI_INT, MPI_SUM, parent->icCommunicator()->communicator());
   MPI_Allreduce(MPI_IN_PLACE, &currTrueNeg, 1, MPI_INT, MPI_SUM, parent->icCommunicator()->communicator());
   MPI_Allreduce(MPI_IN_PLACE, &sumsq, 1, MPI_FLOAT, MPI_SUM, parent->icCommunicator()->communicator());
#endif
   numTotPos += currNumTotPos;
   numTotNeg += currNumTotNeg;
   truePos += currTruePos;
   trueNeg += currTrueNeg;
   progressNumTotPos += currNumTotPos;
   progressNumTotNeg += currNumTotNeg;
   progressTruePos += currTruePos;
   progressTrueNeg += currTrueNeg;
   //Print if need
   float timef = parent->simulationTime();
   if(timef >= nextStatProgress){
      //Update nextStatProgress
      nextStatProgress += statProgressPeriod;
      if (parent->columnId()==0) {
         float totalScore = 50*(float(truePos)/float(numTotPos) + float(trueNeg)/float(numTotNeg));
         float progressScore = 50*(float(progressTruePos)/float(progressNumTotPos) + float(progressTrueNeg)/float(progressNumTotNeg));
         fprintf(stdout, "time:%f  layer:\"%s\"  total:%f%%  progressStep:%f%%  energy:%f\n", timef, name, totalScore, progressScore, sumsq/2);
      }
      //Reset progressStats
      progressNumTotPos = 0;
      progressNumTotNeg = 0;
      progressTruePos = 0;
      progressTrueNeg = 0;
   }
}

int MLPOutputLayer::updateState(double timef, double dt) {
   int status = MLPSigmoidLayer::updateState(timef, dt);
   //Collect stats if needed
   if(statProgressPeriod > 0){
      //TODO add more if statements for different cases
      if(!localTarget){
         if(getLayerLoc()->nf == 1){
            binaryNonlocalStats();
         }
         else{
            multiclassNonlocalStats();
         }
      }
      else{
         if(getLayerLoc()->nf == 1){
            binaryLocalStats();
         }
         else{
            std::cout << "Not implemented\n";
            exit(EXIT_FAILURE);
            //TODO
            //multiclassNonlocalStats();
         }
      }
   }
   //For testing purposes
   pvdata_t * A = getCLayer()->activity->data;
  // for(int ni = 0; ni < getNumNeurons(); ni++){
      //int nExt = kIndexExtended(ni, loc->nx, loc->ny, loc->nf, loc->nb);
  //    std::cout << timef <<":  ni: " << ni << "  A: " << A[ni] << "\n";
  // }
   return PV_SUCCESS;
}

PV::BaseObject * createMLPOutputLayer(char const * name, PV::HyPerCol * hc) { 
   return hc ? new MLPOutputLayer(name, hc) : NULL;
}

}  // end namespace PVMLearning
