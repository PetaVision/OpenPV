#include "BinningLayer.hpp"

namespace PV {

BinningLayer::BinningLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

BinningLayer::BinningLayer() {
   initialize_base();
   // initialize() gets called by subclass's initialize method
}

int BinningLayer::initialize_base() {
   numChannels = 0;
   originalLayerName = NULL;
   originalLayer = NULL;
   delay = 0;
   binMax = 1;
   binMin = 0;
   binSigma = 0;
   return PV_SUCCESS;
}

int BinningLayer::initialize(const char * name, HyPerCol * hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

int BinningLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerLayer::ioParamsFillGroup(ioFlag);
   ioParam_originalLayerName(ioFlag);
   ioParam_binMaxMin(ioFlag);
   ioParam_delay(ioFlag);
   ioParam_binSigma(ioFlag);
   return status;
}

void BinningLayer::ioParam_originalLayerName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "originalLayerName", &originalLayerName);
   assert(originalLayerName);
   if (ioFlag==PARAMS_IO_READ && originalLayerName[0]=='\0') {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: originalLayerName must be set.\n",
                 parent->parameters()->groupKeywordFromName(name), name);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
}

void BinningLayer::ioParam_binMaxMin(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "binMax", &binMax, binMax);
   parent->ioParamValue(ioFlag, name, "binMin", &binMin, binMin);
   if(ioFlag == PARAMS_IO_READ && binMax <= binMin){
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: binMax (%f) must be greater than binMin (%f).\n",
            parent->parameters()->groupKeywordFromName(name), name, binMax, binMin);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
}

void BinningLayer::ioParam_binSigma(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "binSigma", &binSigma, binSigma);
}

void BinningLayer::ioParam_delay(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "delay", &delay, delay);
}

//TODO read params for gaussian over features

int BinningLayer::communicateInitInfo() {
   int status = HyPerLayer::communicateInitInfo();
   originalLayer = parent->getLayerFromName(originalLayerName);
   if (originalLayer==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: originalLayerName \"%s\" is not a layer in the HyPerCol.\n",
                 parent->parameters()->groupKeywordFromName(name), name, originalLayerName);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   originalLayer->synchronizeMarginWidth(this);
   const PVLayerLoc * srcLoc = originalLayer->getLayerLoc();
   const PVLayerLoc * loc = getLayerLoc();
   assert(srcLoc != NULL && loc != NULL);
   if (srcLoc->nxGlobal != loc->nxGlobal || srcLoc->nyGlobal != loc->nyGlobal) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: originalLayerName \"%s\" does not have the same dimensions.\n",
                 parent->parameters()->groupKeywordFromName(name), name, originalLayerName);
         fprintf(stderr, "    original (nx=%d, ny=%d) versus (nx=%d, ny=%d)\n",
                 srcLoc->nxGlobal, srcLoc->nyGlobal, loc->nxGlobal, loc->nyGlobal);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   if(srcLoc->nf != 1){
      fprintf(stderr, "%s \"%s\" error: originalLayerName \"%s\" can only have 1 feature.\n",
         parent->parameters()->groupKeywordFromName(name), name, originalLayerName);
   }
   assert(srcLoc->nx==loc->nx && srcLoc->ny==loc->ny);
   return status;
}

int BinningLayer::requireMarginWidth(int marginWidthNeeded, int * marginWidthResult) {
   HyPerLayer::requireMarginWidth(marginWidthNeeded, marginWidthResult);
   assert(*marginWidthResult >= marginWidthNeeded);
   originalLayer->requireMarginWidth(marginWidthNeeded, marginWidthResult);
   assert(*marginWidthResult>=marginWidthNeeded);
   return PV_SUCCESS;
}

int BinningLayer::allocateDataStructures() {
   int status = HyPerLayer::allocateDataStructures();
   return status;
}

int BinningLayer::allocateV(){
   //Allocate V does nothing since binning does not need a V layer
   clayer->V = NULL;
   return PV_SUCCESS;
}

int BinningLayer::initializeV() {
   assert(getV() == NULL);
   return PV_SUCCESS;
}

int BinningLayer::initializeActivity() {
   return PV_SUCCESS;
}

int BinningLayer::updateState(double timef, double dt) {
   int status;
   assert(GSyn==NULL);
   pvdata_t * gSynHead = NULL;

   status = doUpdateState(timef, dt, originalLayer->getLayerLoc(), getLayerLoc(), originalLayer->getLayerData(delay), getActivity(), binMax, binMin);
   if(status == PV_SUCCESS) status = updateActiveIndices();
   return status;
}

int BinningLayer::doUpdateState(double timed, double dt, const PVLayerLoc * origLoc, const PVLayerLoc * currLoc, const pvdata_t * origData, pvdata_t * currA, float binMax, float binMin) {
   int status = PV_SUCCESS;
   update_timer->start();
   int numBins = currLoc->nf;

   int nx = currLoc->nx;
   int ny = currLoc->ny;
   //Check that both nb are the same
   assert(origLoc->nb == currLoc->nb);
   assert(origLoc->nf == 1);
   int nb = origLoc->nb;
   float binRange = binMax - binMin;
   float stepSize = float(binRange)/numBins;
   for (int iY = 0; iY < (ny+2*nb); iY++){
      for (int iX = 0; iX < (nx+2*nb); iX++){
         int origIdx = kIndex(iX, iY, 0, nx+2*nb, ny+2*nb, origLoc->nf);
         float inVal = origData[origIdx];
         //A sigma of zero means only the centered bin value should get input
         int featureIdx = round((inVal-binMin)/stepSize);
         for(int iF = 0; iF < numBins; iF++){
            if(binSigma == 0){
               int currIdx = kIndex(iX, iY, iF, nx+2*nb, ny+2*nb, numBins);
               if(iF == featureIdx){
                  currA[currIdx] = 1;
               }
               //Resetting value
               else{
                  currA[currIdx] = 0;
               }
            }
            else{
               //Calculate center value for featureIdx (the bin that the value belongs to without a sigma) is binning
               float mean = featureIdx * stepSize + (stepSize/2);
               //Possible bins
               int intSigma = ceil(binSigma);
               int currIdx = kIndex(iX, iY, iF, nx+2*nb, ny+2*nb, numBins);
               if(iF >= featureIdx-intSigma && iF <= featureIdx+intSigma){
                  //Get center of that aBin for the x pos of the normal dist
                  float xVal = iF * stepSize + (stepSize/2);
                  //Calculate normal dist
                  float outVal = calcNormDist(xVal, mean, binSigma);
                  //Put into activity buffer
                  currA[currIdx] = outVal;
               }
               //Resetting value
               else{
                  currA[currIdx] = 0;
               }
            }
         }
      }
   }
   update_timer->stop();
   return status;
}

float BinningLayer::calcNormDist(float xVal, float mean, float sigma){
   return (float(1)/(sigma*(sqrt(2*PI))))*exp(-(pow(xVal-mean, 2)/(2*pow(sigma, 2))));
}

BinningLayer::~BinningLayer() {
   free(originalLayerName);
   clayer->V = NULL;
}

} /* namespace PV */
