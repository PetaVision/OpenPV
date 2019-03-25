#include "BinningLayer.hpp"

namespace PV {

BinningLayer::BinningLayer(const char *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

BinningLayer::BinningLayer() {
   initialize_base();
   // initialize() gets called by subclass's initialize method
}

int BinningLayer::initialize_base() {
   numChannels = 0;
   return PV_SUCCESS;
}

int BinningLayer::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

int BinningLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerLayer::ioParamsFillGroup(ioFlag);
   ioParam_originalLayerName(ioFlag);
   ioParam_binMin(ioFlag);
   ioParam_binMax(ioFlag);
   ioParam_delay(ioFlag);
   ioParam_binSigma(ioFlag);
   ioParam_zeroNeg(ioFlag);
   ioParam_zeroDCR(ioFlag);
   return status;
}

void BinningLayer::ioParam_originalLayerName(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamStringRequired(
         ioFlag, name, "originalLayerName", &mOriginalLayerName);
   assert(mOriginalLayerName);
   if (ioFlag == PARAMS_IO_READ && mOriginalLayerName[0] == '\0') {
      if (parent->columnId() == 0) {
         ErrorLog().printf("%s: originalLayerName must be set.\n", getDescription_c());
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
}

void BinningLayer::ioParam_binMin(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "binMin", &mBinMin, mBinMin);
}

void BinningLayer::ioParam_binMax(enum ParamsIOFlag ioFlag) {
   pvAssert(!parent->parameters()->presentAndNotBeenRead(name, "binMin"));
   parent->parameters()->ioParamValue(ioFlag, name, "binMax", &mBinMax, mBinMax);
   if (ioFlag == PARAMS_IO_READ && mBinMax <= mBinMin) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "%s: binMax (%f) must be greater than binMin (%f).\n",
               getDescription_c(),
               (double)mBinMax,
               (double)mBinMin);
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
}

void BinningLayer::ioParam_binSigma(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "binSigma", &mBinSigma, mBinSigma);
}

void BinningLayer::ioParam_delay(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "delay", &mDelay, mDelay);
}

void BinningLayer::ioParam_zeroNeg(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "zeroNeg", &mZeroNeg, mZeroNeg);
}

void BinningLayer::ioParam_zeroDCR(enum ParamsIOFlag ioFlag) {
   parent->parameters()->ioParamValue(ioFlag, name, "zeroDCR", &mZeroDCR, mZeroDCR);
}

// TODO read params for gaussian over features

Response::Status
BinningLayer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = HyPerLayer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   mOriginalLayer = message->lookup<HyPerLayer>(std::string(mOriginalLayerName));
   if (mOriginalLayer == NULL) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "%s: originalLayerName \"%s\" is not a layer in the HyPerCol.\n",
               getDescription_c(),
               mOriginalLayerName);
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   if (mOriginalLayer->getInitInfoCommunicatedFlag() == false) {
      return Response::POSTPONE;
   }
   mOriginalLayer->synchronizeMarginWidth(this);
   this->synchronizeMarginWidth(mOriginalLayer);
   const PVLayerLoc *srcLoc = mOriginalLayer->getLayerLoc();
   const PVLayerLoc *loc    = getLayerLoc();
   assert(srcLoc != NULL && loc != NULL);
   if (srcLoc->nxGlobal != loc->nxGlobal || srcLoc->nyGlobal != loc->nyGlobal) {
      if (parent->columnId() == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf(
               "%s: originalLayerName \"%s\" does not have the same dimensions.\n",
               getDescription_c(),
               mOriginalLayerName);
         errorMessage.printf(
               "    original (nx=%d, ny=%d) versus (nx=%d, ny=%d)\n",
               srcLoc->nxGlobal,
               srcLoc->nyGlobal,
               loc->nxGlobal,
               loc->nyGlobal);
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   if (srcLoc->nf != 1) {
      ErrorLog().printf(
            "%s: originalLayerName \"%s\" can only have 1 feature.\n",
            getDescription_c(),
            mOriginalLayerName);
   }
   assert(srcLoc->nx == loc->nx && srcLoc->ny == loc->ny);
   return Response::SUCCESS;
}

Response::Status BinningLayer::allocateDataStructures() {
   return HyPerLayer::allocateDataStructures();
}

void BinningLayer::allocateV() {
   // Allocate V does nothing since binning does not need a V layer
   clayer->V = NULL;
}

void BinningLayer::initializeV() { assert(getV() == NULL); }

void BinningLayer::initializeActivity() {}

Response::Status BinningLayer::updateState(double simTime, double dt) {
   PVLayerLoc const *origLoc = mOriginalLayer->getLayerLoc();
   PVLayerLoc const *currLoc = getLayerLoc();

   pvAssert(origLoc->nx == currLoc->nx);
   pvAssert(origLoc->ny == currLoc->ny);
   pvAssert(origLoc->nf == 1);
   int nx = currLoc->nx;
   int ny = currLoc->ny;

   pvAssert(origLoc->halo.lt == currLoc->halo.lt);
   pvAssert(origLoc->halo.rt == currLoc->halo.rt);
   pvAssert(origLoc->halo.dn == currLoc->halo.dn);
   pvAssert(origLoc->halo.up == currLoc->halo.up);
   PVHalo const *halo = &origLoc->halo;

   int numBins    = currLoc->nf;
   float binRange = mBinMax - mBinMin;
   float stepSize = float(binRange) / numBins;

   int const nxExt = origLoc->nx + origLoc->halo.lt + origLoc->halo.rt;
   int const nyExt = origLoc->ny + origLoc->halo.dn + origLoc->halo.up;

   float const *origData = mOriginalLayer->getLayerData(mDelay);
   float *currActivity   = getActivity();

   int nbatch = currLoc->nbatch;
   pvAssert(origLoc->nbatch == currLoc->nbatch);
   for (int b = 0; b < nbatch; b++) {
      const float *origDataBatch = origData + b * nxExt * nyExt * origLoc->nf;
      float *currABatch          = currActivity + b * nxExt * nyExt * currLoc->nf;

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for (int kExt = 0; kExt < nxExt * nyExt * currLoc->nf; kExt++) {
         int iX = kxPos(kExt, nxExt, nyExt, currLoc->nf);
         int iY = kyPos(kExt, nxExt, nyExt, currLoc->nf);
         int iF = featureIndex(kExt, nxExt, nyExt, currLoc->nf);

         int origIndex = kIndex(iX, iY, 0, nxExt, nyExt, 1);
         float inVal   = origDataBatch[origIndex];
         // If inVal is out of bounds in either binMax or binMin, set the value to be the maximum
         // or minimum val
         inVal = inVal < mBinMin ? mBinMin : inVal > mBinMax ? mBinMax : inVal;

         int const outOfBinValue = mZeroNeg ? 0 : -1;
         if (mZeroDCR && inVal == 0) { // do-not-care region
            currABatch[kExt] = outOfBinValue;
         }
         else {
            // A sigma of zero means only the centered bin value should get input
            if (mBinSigma == 0) {
               int featureIdx = std::floor((inVal - mBinMin) / stepSize);
               if (featureIdx >= numBins) {
                  featureIdx = numBins - 1;
               }
               currABatch[kExt] = iF == featureIdx ? 1 : outOfBinValue;
            }
            // sigma>0 means bin values have Gaussian decay as bins move farther from inVal.
            // The width of the Gaussian is binValue multiplied by the bin width.
            else {
               float binCenter  = ((float)iF + 0.5f) * stepSize;
               currABatch[kExt] = calcGaussian(inVal - binCenter, mBinSigma * stepSize);
            }
         }
      }
   }
   return Response::SUCCESS;
}

float BinningLayer::calcGaussian(float x, float sigma) {
   return std::exp(-x * x / (2 * sigma * sigma));
}

BinningLayer::~BinningLayer() {
   free(mOriginalLayerName);
   clayer->V = NULL;
}

} /* namespace PV */
