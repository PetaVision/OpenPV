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
   originalLayerName = NULL;
   originalLayer = NULL;
   delay = 0;
   binMax = 1;
   binMin = 0;
   binSigma = 0;
   return PV_SUCCESS;
}

int BinningLayer::initialize(const char * name, HyPerCol * hc) {
   int status = HyPerLayer::initialize(name, hc, 1);
   return status;
}

int BinningLayer::setParams(PVParams * params) {
   int status = HyPerLayer::setParams(params);
   readOriginalLayerName(params);
   readBinMaxMin(params);
   readDelay(params);
   readBinSigma(params);
   return status;
}

void BinningLayer::readOriginalLayerName(PVParams * params) {
   const char * original_layer_name = params->stringValue(name, "originalLayerName");
   if (original_layer_name==NULL || original_layer_name[0]=='\0') {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: originalLayerName must be set.\n",
                 parent->parameters()->groupKeywordFromName(name), name);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   originalLayerName = strdup(original_layer_name);
   if (originalLayerName==NULL) {
      fprintf(stderr, "%s \"%s\" error: rank %d process unable to copy originalLayerName \"%s\": %s\n",
              parent->parameters()->groupKeywordFromName(name), name, parent->columnId(), original_layer_name, strerror(errno));
      exit(EXIT_FAILURE);
   }
}

void BinningLayer::readBinMaxMin(PVParams * params) {
   binMax = params->value(name, "binMax", binMax);
   binMin = params->value(name, "binMin", binMin);
   if(binMax <= binMin){
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: binMax (%f) must be greater than binMin (%f).\n",
            parent->parameters()->groupKeywordFromName(name), name, binMax, binMin);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
}

void BinningLayer::readBinSigma(PVParams * params) {
   binSigma = params->value(name, "binSigma", binSigma);
}

void BinningLayer::readDelay(PVParams * params) {
   delay = (int) params->value(name, "delay", delay);
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

int BinningLayer::initializeState() {
   int status = PV_SUCCESS;
   PVParams * params = parent->parameters();
   assert(!params->presentAndNotBeenRead(name, "restart"));
   // readRestart(params);
   if( restartFlag ) {
      double timef;
      status = readState(&timef);
   }
   else {
      //status = setActivity();
      //if (status == PV_SUCCESS) status = updateActiveIndices();
   }
   return status;
}

int BinningLayer::updateState(double timef, double dt) {
   int status;
   pvdata_t * gSynHead = GSyn==NULL ? NULL : GSyn[0];
   assert(getNumChannels() == 1);

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
