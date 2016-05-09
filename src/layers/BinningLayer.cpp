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
   zeroNeg = true;
   zeroDCR = false;
   normalDist = true;
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
   ioParam_zeroNeg(ioFlag);
   ioParam_zeroDCR(ioFlag);
   ioParam_normalDist(ioFlag);
   return status;
}

void BinningLayer::ioParam_originalLayerName(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "originalLayerName", &originalLayerName);
   assert(originalLayerName);
   if (ioFlag==PARAMS_IO_READ && originalLayerName[0]=='\0') {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: originalLayerName must be set.\n",
                 getKeyword(), name);
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
            getKeyword(), name, binMax, binMin);
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

void BinningLayer::ioParam_zeroNeg(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "zeroNeg", &zeroNeg, zeroNeg);
}

void BinningLayer::ioParam_zeroDCR(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "zeroDCR", &zeroDCR, zeroDCR);
}

void BinningLayer::ioParam_normalDist(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "normalDist", &normalDist, normalDist);
}

//TODO read params for gaussian over features

int BinningLayer::communicateInitInfo() {
   int status = HyPerLayer::communicateInitInfo();
   originalLayer = parent->getLayerFromName(originalLayerName);
   if (originalLayer==NULL) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: originalLayerName \"%s\" is not a layer in the HyPerCol.\n",
                 getKeyword(), name, originalLayerName);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   if (originalLayer->getInitInfoCommunicatedFlag()==false) {
      return PV_POSTPONE;
   }
   originalLayer->synchronizeMarginWidth(this);
   this->synchronizeMarginWidth(originalLayer);
   const PVLayerLoc * srcLoc = originalLayer->getLayerLoc();
   const PVLayerLoc * loc = getLayerLoc();
   assert(srcLoc != NULL && loc != NULL);
   if (srcLoc->nxGlobal != loc->nxGlobal || srcLoc->nyGlobal != loc->nyGlobal) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: originalLayerName \"%s\" does not have the same dimensions.\n",
                 getKeyword(), name, originalLayerName);
         fprintf(stderr, "    original (nx=%d, ny=%d) versus (nx=%d, ny=%d)\n",
                 srcLoc->nxGlobal, srcLoc->nyGlobal, loc->nxGlobal, loc->nyGlobal);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   if(srcLoc->nf != 1){
      fprintf(stderr, "%s \"%s\" error: originalLayerName \"%s\" can only have 1 feature.\n",
         getKeyword(), name, originalLayerName);
   }
   assert(srcLoc->nx==loc->nx && srcLoc->ny==loc->ny);
   return status;
}

int BinningLayer::requireMarginWidth(int marginWidthNeeded, int * marginWidthResult, char axis) {
   HyPerLayer::requireMarginWidth(marginWidthNeeded, marginWidthResult, axis);
   assert(*marginWidthResult >= marginWidthNeeded);
   // The code below is handled by the synchronizeMarginWidth call in communicateInitInfo
   // originalLayer->requireMarginWidth(marginWidthNeeded, marginWidthResult, axis);
   // assert(*marginWidthResult>=marginWidthNeeded);
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
   return status;
}

int BinningLayer::doUpdateState(double timed, double dt, const PVLayerLoc * origLoc, const PVLayerLoc * currLoc, const pvdata_t * origData, pvdata_t * currA, float binMax, float binMin) {
   int status = PV_SUCCESS;
   //update_timer->start();
   int numBins = currLoc->nf;

   int nx = currLoc->nx;
   int ny = currLoc->ny;
   //Check that both nb are the same
   assert(origLoc->halo.lt == currLoc->halo.lt &&
          origLoc->halo.rt == currLoc->halo.rt &&
          origLoc->halo.dn == currLoc->halo.dn &&
          origLoc->halo.up == currLoc->halo.up);
   assert(origLoc->nf == 1);
   PVHalo const * halo = &origLoc->halo;
   float binRange = binMax - binMin;
   float stepSize = float(binRange)/numBins;
   int nbatch = currLoc->nbatch;


   
   for(int b = 0; b < nbatch; b++){
      const pvdata_t * origDataBatch = origData + b * (origLoc->nx + origLoc->halo.lt + origLoc->halo.rt) * (origLoc->ny + origLoc->halo.up + origLoc->halo.dn) * origLoc->nf;
      pvdata_t * currABatch = currA + b * (currLoc->nx + currLoc->halo.lt + currLoc->halo.rt) * (currLoc->ny + currLoc->halo.up + currLoc->halo.dn) * currLoc->nf;

      // each y value specifies a different target so ok to thread here (sum, sumsq are defined inside loop)
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for (int iY = 0; iY < (ny+halo->dn+halo->up); iY++){
         for (int iX = 0; iX < (nx+halo->lt+halo->rt); iX++){
            int origIdx = kIndex(iX, iY, 0, nx+halo->lt+halo->rt, ny+halo->dn+halo->up, origLoc->nf);
            float inVal = origDataBatch[origIdx];
            //If inVal is out of bounds in either binMax or binMin, set the value to be the maximum or minimum val
            if(inVal < binMin){
               inVal = binMin;
            }
            if(inVal > binMax){
               inVal = binMax;
            }

            if(zeroDCR && inVal == 0){
               for(int iF = 0; iF < numBins; iF++){
                  int currIdx = kIndex(iX, iY, iF, nx+halo->lt+halo->rt, ny+halo->dn+halo->up, numBins);
                  currABatch[currIdx] = 0;
               }
            }
            else{
               //A sigma of zero means only the centered bin value should get input
               int featureIdx = round((inVal-binMin)/stepSize);

               for(int iF = 0; iF < numBins; iF++){
                  if(binSigma == 0){
                     int currIdx = kIndex(iX, iY, iF, nx+halo->lt+halo->rt, ny+halo->dn+halo->up, numBins);
                     if(iF == featureIdx){
                        currABatch[currIdx] = 1;
                     }
                     //Resetting value
                     else{
                        if(zeroNeg){
                           currABatch[currIdx] = 0;
                        }
                        else{
                           currABatch[currIdx] = -1;
                        }
                     }
                  }
                  else{
                     //Calculate center value for featureIdx (the bin that the value belongs to without a sigma) is binning
                     float mean;
                     if(normalDist){
                        mean = featureIdx * stepSize + (stepSize/2);
                     }
                     else{
                        mean = featureIdx;
                     }
                     //Possible bins
                     int intSigma = ceil(binSigma);
                     int currIdx = kIndex(iX, iY, iF, nx+halo->lt+halo->rt, ny+halo->dn+halo->up, numBins);
                     if(iF >= featureIdx-intSigma && iF <= featureIdx+intSigma){
                        //Get center of that aBin for the x pos of the normal dist
                        float xVal;
                        if(normalDist){
                           xVal = iF * stepSize + (stepSize/2);
                        }
                        else{
                           xVal = iF;
                        }
                        //Calculate normal dist
                        float outVal = calcNormDist(xVal, mean, binSigma);
                        //Put into activity buffer
                        currABatch[currIdx] = outVal;
                     }
                     //Resetting value
                     else{
                        if(zeroNeg){
                           currABatch[currIdx] = 0;
                        }
                        else{
                           currABatch[currIdx] = -1;
                        }
                     }
                  }
               }
            }
         }
      }
   }
   //update_timer->stop();
   return status;
}

float BinningLayer::calcNormDist(float xVal, float mean, float sigma){
   if(normalDist){
      return (float(1)/(sigma*(sqrt(2*PI))))*exp(-(pow(xVal-mean, 2)/(2*pow(sigma, 2))));
   }
   else{
      return exp(-(pow(xVal-mean, 2)/(2*pow((sigma/2), 2))));
   }
}

BinningLayer::~BinningLayer() {
   free(originalLayerName);
   clayer->V = NULL;
}

BaseObject * createBinningLayer(char const * name, HyPerCol * hc) {
   return hc ? new BinningLayer(name, hc) : NULL;
}

} /* namespace PV */
