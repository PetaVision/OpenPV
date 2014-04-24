/*
 * MLPOutputLayer.cpp
 * Author: slundquist
 */

#include "MLPOutputLayer.hpp"

namespace PV {
MLPOutputLayer::MLPOutputLayer(){
   initialize_base();
}
MLPOutputLayer::MLPOutputLayer(const char * name, HyPerCol * hc)
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
   statProgressPeriod = 0; //Never print progress
   nextStatProgress = 0;
   return PV_SUCCESS;
}

int MLPOutputLayer::initialize(const char * name, HyPerCol * hc)
{
   int status = SigmoidLayer::initialize(name, hc);
   if(statProgressPeriod > 0){
      nextStatProgress = hc->getStartTime() + statProgressPeriod;
   }
   return status;
}

//TODO checkpoint stats and nextStatProgress

int MLPOutputLayer::communicateInitInfo(){
   int status = SigmoidLayer::communicateInitInfo();
   if(statProgressPeriod > 0){
      gtLayer = parent->getLayerFromName(gtLayername);
      if (gtLayer==NULL) {
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: gtLayername \"%s\" is not a layer in the HyPerCol.\n",
                    parent->parameters()->groupKeywordFromName(name), name, gtLayername);
         }
#if PV_USE_MPI
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
                    parent->parameters()->groupKeywordFromName(name), name, gtLayername);
            fprintf(stderr, "    original (nx=%d, ny=%d, nf=%d) versus (nx=%d, ny=%d, nf=%d)\n",
                    srcLoc->nxGlobal, srcLoc->nyGlobal, srcLoc->nf, loc->nxGlobal, loc->nyGlobal, loc->nf);
         }
#if PV_USE_MPI
         MPI_Barrier(parent->icCommunicator()->communicator());
#endif
         exit(EXIT_FAILURE);
      }
   }
   return(status);
}

int MLPOutputLayer::allocateDataStructures() {
   int status = SigmoidLayer::allocateDataStructures();
   //Allocate buffer size of this layer's nf (number of output classes)
   int nf = getLayerLoc()->nf;
   if(!localTarget){
      classBuffer = (pvdata_t*) malloc(nf * sizeof(pvdata_t));
   }
   //Grab gtLayername if it exists

   return status;
}

int MLPOutputLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = SigmoidLayer::ioParamsFillGroup(ioFlag);
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

int MLPOutputLayer::updateState(double timef, double dt) {
   int status = SigmoidLayer::updateState(timef, dt);
   //If not local, find mean of all output nodes and set to all nodes
   //TODO is mean the right way to do this?
   const PVLayerLoc * loc = getLayerLoc();
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int numNeurons = getNumNeurons();
   pvdata_t * A = getCLayer()->activity->data;
   if(!localTarget){
      assert(classBuffer);
      //Clear classBuffer
      for(int i = 0; i < nf; i++){
         classBuffer[i] = 0;
      }
      //Only go through restricted
      for(int ni = 0; ni < numNeurons; ni++){
         int nExt = kIndexExtended(ni, nx, ny, nf, loc->nb);
         int fi = featureIndex(nExt, nx+2*loc->nb, ny+2*loc->nb, nf);
         //Sum over x and y direction
         classBuffer[fi] += A[nExt];
      }
      //Normalize classBuffer to find mean
      for(int i = 0; i < nf; i++){
         classBuffer[i] /= nx*ny;
      }

      //Reduce all classBuffers through a mean
#ifdef PV_USE_MPI
      MPI_Allreduce(MPI_IN_PLACE, classBuffer, nf, MPI_FLOAT, MPI_SUM, parent->icCommunicator()->communicator());
      //Normalize classBuffer across processors
      for(int i = 0; i < nf; i++){
         classBuffer[i] /= parent->icCommunicator()->commSize();
      }
#endif // PV_USE_MPI

      //Put new classBuffer into activity
      //Only go through restricted
      for(int ni = 0; ni < numNeurons; ni++){
         int nExt = kIndexExtended(ni, nx, ny, nf, loc->nb);
         int fi = featureIndex(nExt, nx+2*loc->nb, ny+2*loc->nb, nf);
         A[nExt] = classBuffer[fi];
      }
   }

   //Collect stats if needed
   if(statProgressPeriod > 0){
      pvdata_t * gtA = gtLayer->getCLayer()->activity->data;
      int currNumRight = 0;
      int currNumWrong = 0;
      for(int ni = 0; ni < numNeurons; ni++){ 
         int nExt = kIndexExtended(ni, nx, ny, nf, loc->nb);
         if(round(A[nExt]) == gtA[nExt]){
            currNumRight++;
         }
         else{
            currNumWrong++;
         }
      }
#ifdef PV_USE_MPI
      MPI_Allreduce(MPI_IN_PLACE, &currNumRight, 1, MPI_INT, MPI_SUM, parent->icCommunicator()->communicator());
      MPI_Allreduce(MPI_IN_PLACE, &currNumWrong, 1, MPI_INT, MPI_SUM, parent->icCommunicator()->communicator());
#endif // PV_USE_MPI
      numRight += currNumRight;
      numWrong += currNumWrong;

      if(timef >= nextStatProgress){
         //Update nextStatProgress
         nextStatProgress += statProgressPeriod;
         if (parent->columnId()==0) {
            fprintf(stdout, "%s stat output: %f%% correct\n", name, 100*float(numRight)/float(numRight+numWrong));
         }
      }
      //Check progress period and print out score so far
   }

   //For testing purposes
   //for(int ni = 0; ni < getNumNeurons(); ni++){
   //   int nExt = kIndexExtended(ni, nx, ny, nf, loc->nb);
   //   std::cout << timef <<":  ni: " << ni << "  A: " << A[ni] << "\n";
   //}
   return PV_SUCCESS;
}

}
