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
   delay      = 0;
   binMax     = 1;
   binMin     = 0;
   binSigma   = 0;
   zeroNeg    = true;
   zeroDCR    = false;
   normalDist = true;
   return PV_SUCCESS;
}

int BinningLayer::initialize(const char *name, HyPerCol *hc) {
   int status = HyPerLayer::initialize(name, hc);
   return status;
}

void BinningLayer::setObserverTable() {
   HyPerLayer::setObserverTable();
   auto *originalLayerNameParam = createOriginalLayerNameParam();
   if (originalLayerNameParam) {
      addUniqueComponent(originalLayerNameParam->getDescription(), originalLayerNameParam);
   }
}

LayerInputBuffer *BinningLayer::createLayerInput() { return nullptr; }

InternalStateBuffer *BinningLayer::createInternalState() { return nullptr; }

OriginalLayerNameParam *BinningLayer::createOriginalLayerNameParam() {
   return new OriginalLayerNameParam(name, parent);
}

int BinningLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerLayer::ioParamsFillGroup(ioFlag);
   ioParam_binMaxMin(ioFlag);
   ioParam_delay(ioFlag);
   ioParam_binSigma(ioFlag);
   ioParam_zeroNeg(ioFlag);
   ioParam_zeroDCR(ioFlag);
   ioParam_normalDist(ioFlag);
   return status;
}

void BinningLayer::ioParam_binMaxMin(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "binMax", &binMax, binMax);
   parameters()->ioParamValue(ioFlag, name, "binMin", &binMin, binMin);
   if (ioFlag == PARAMS_IO_READ && binMax <= binMin) {
      if (parent->getCommunicator()->commRank() == 0) {
         ErrorLog().printf(
               "%s: binMax (%f) must be greater than binMin (%f).\n",
               getDescription_c(),
               (double)binMax,
               (double)binMin);
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
}

void BinningLayer::ioParam_binSigma(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "binSigma", &binSigma, binSigma);
}

void BinningLayer::ioParam_delay(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "delay", &delay, delay);
}

void BinningLayer::ioParam_zeroNeg(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "zeroNeg", &zeroNeg, zeroNeg);
}

void BinningLayer::ioParam_zeroDCR(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "zeroDCR", &zeroDCR, zeroDCR);
}

void BinningLayer::ioParam_normalDist(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValue(ioFlag, name, "normalDist", &normalDist, normalDist);
}

// TODO read params for gaussian over features

Response::Status
BinningLayer::communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = HyPerLayer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }

   setOriginalLayer();
   pvAssert(mOriginalLayer);

   if (mOriginalLayer->getInitInfoCommunicatedFlag() == false) {
      return Response::POSTPONE;
   }
   mOriginalLayer->synchronizeMarginWidth(this);
   this->synchronizeMarginWidth(mOriginalLayer);
   const PVLayerLoc *srcLoc = mOriginalLayer->getLayerLoc();
   const PVLayerLoc *loc    = getLayerLoc();
   assert(srcLoc != nullptr && loc != nullptr);
   if (srcLoc->nxGlobal != loc->nxGlobal || srcLoc->nyGlobal != loc->nyGlobal) {
      if (parent->getCommunicator()->commRank() == 0) {
         ErrorLog(errorMessage);
         errorMessage.printf(
               "%s: originalLayerName \"%s\" does not have the same dimensions.\n",
               getDescription_c(),
               mOriginalLayer->getName());
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
            mOriginalLayer->getName());
   }
   assert(srcLoc->nx == loc->nx && srcLoc->ny == loc->ny);
   return Response::SUCCESS;
}

void BinningLayer::setOriginalLayer() {
   auto *originalLayerNameParam = getComponentByType<OriginalLayerNameParam>();
   pvAssert(originalLayerNameParam);

   ComponentBasedObject *originalObject = nullptr;
   try {
      originalObject = originalLayerNameParam->findLinkedObject(mObserverTable);
   } catch (std::invalid_argument &e) {
      Fatal().printf("%s: %s\n", getDescription_c(), e.what());
   }
   pvAssert(originalObject);

   mOriginalLayer = dynamic_cast<HyPerLayer *>(originalObject);
   if (mOriginalLayer == nullptr) {
      if (parent->getCommunicator()->globalCommRank() == 0) {
         ErrorLog().printf(
               "%s: originalLayerName \"%s\" is not a layer in the HyPerCol.\n",
               getDescription_c(),
               originalObject->getName());
      }
      MPI_Barrier(parent->getCommunicator()->globalCommunicator());
      exit(EXIT_FAILURE);
   }
}

Response::Status BinningLayer::allocateDataStructures() {
   return HyPerLayer::allocateDataStructures();
}

void BinningLayer::initializeActivity() {}

Response::Status BinningLayer::updateState(double timef, double dt) {
   assert(mLayerInput == NULL);

   doUpdateState(
         timef,
         dt,
         mOriginalLayer->getLayerLoc(),
         getLayerLoc(),
         mOriginalLayer->getLayerData(delay),
         getActivity(),
         binMax,
         binMin);
   return Response::SUCCESS;
}

void BinningLayer::doUpdateState(
      double timed,
      double dt,
      const PVLayerLoc *origLoc,
      const PVLayerLoc *currLoc,
      const float *origData,
      float *currA,
      float binMax,
      float binMin) {
   int status  = PV_SUCCESS;
   int numBins = currLoc->nf;

   int nx = currLoc->nx;
   int ny = currLoc->ny;
   // Check that both nb are the same
   assert(
         origLoc->halo.lt == currLoc->halo.lt && origLoc->halo.rt == currLoc->halo.rt
         && origLoc->halo.dn == currLoc->halo.dn
         && origLoc->halo.up == currLoc->halo.up);
   assert(origLoc->nf == 1);
   PVHalo const *halo = &origLoc->halo;
   float binRange     = binMax - binMin;
   float stepSize     = float(binRange) / numBins;
   int nbatch         = currLoc->nbatch;

   for (int b = 0; b < nbatch; b++) {
      const float *origDataBatch = origData
                                   + b * (origLoc->nx + origLoc->halo.lt + origLoc->halo.rt)
                                           * (origLoc->ny + origLoc->halo.up + origLoc->halo.dn)
                                           * origLoc->nf;
      float *currABatch = currA
                          + b * (currLoc->nx + currLoc->halo.lt + currLoc->halo.rt)
                                  * (currLoc->ny + currLoc->halo.up + currLoc->halo.dn)
                                  * currLoc->nf;

// each y value specifies a different target so ok to thread here (sum, sumsq are defined inside
// loop)
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for (int iY = 0; iY < (ny + halo->dn + halo->up); iY++) {
         for (int iX = 0; iX < (nx + halo->lt + halo->rt); iX++) {
            int origIdx = kIndex(
                  iX, iY, 0, nx + halo->lt + halo->rt, ny + halo->dn + halo->up, origLoc->nf);
            float inVal = origDataBatch[origIdx];
            // If inVal is out of bounds in either binMax or binMin, set the value to be the maximum
            // or minimum val
            if (inVal < binMin) {
               inVal = binMin;
            }
            if (inVal > binMax) {
               inVal = binMax;
            }

            if (zeroDCR && inVal == 0) {
               for (int iF = 0; iF < numBins; iF++) {
                  int currIdx = kIndex(
                        iX, iY, iF, nx + halo->lt + halo->rt, ny + halo->dn + halo->up, numBins);
                  currABatch[currIdx] = 0;
               }
            }
            else {
               // A sigma of zero means only the centered bin value should get input
               int featureIdx = round((inVal - binMin) / stepSize);

               for (int iF = 0; iF < numBins; iF++) {
                  if (binSigma == 0) {
                     int currIdx = kIndex(
                           iX, iY, iF, nx + halo->lt + halo->rt, ny + halo->dn + halo->up, numBins);
                     if (iF == featureIdx) {
                        currABatch[currIdx] = 1;
                     }
                     // Resetting value
                     else {
                        if (zeroNeg) {
                           currABatch[currIdx] = 0;
                        }
                        else {
                           currABatch[currIdx] = -1;
                        }
                     }
                  }
                  else {
                     // Calculate center value for featureIdx (the bin that the value belongs to
                     // without a sigma) is binning
                     float mean;
                     if (normalDist) {
                        mean = featureIdx * stepSize + (stepSize / 2);
                     }
                     else {
                        mean = featureIdx;
                     }
                     // Possible bins
                     int intSigma = ceil(binSigma);
                     int currIdx  = kIndex(
                           iX, iY, iF, nx + halo->lt + halo->rt, ny + halo->dn + halo->up, numBins);
                     if (iF >= featureIdx - intSigma && iF <= featureIdx + intSigma) {
                        // Get center of that aBin for the x pos of the normal dist
                        float xVal;
                        if (normalDist) {
                           xVal = iF * stepSize + (stepSize / 2);
                        }
                        else {
                           xVal = iF;
                        }
                        // Calculate normal dist
                        float outVal = calcNormDist(xVal, mean, binSigma);
                        // Put into activity buffer
                        currABatch[currIdx] = outVal;
                     }
                     // Resetting value
                     else {
                        if (zeroNeg) {
                           currABatch[currIdx] = 0;
                        }
                        else {
                           currABatch[currIdx] = -1;
                        }
                     }
                  }
               }
            }
         }
      }
   }
}

float BinningLayer::calcNormDist(float xVal, float mean, float sigma) {
   if (normalDist) {
      return 1.0f / (sigma * (sqrtf(2.0f * (float)PI)))
             * expf(-(powf(xVal - mean, 2.0f) / (2.0f * powf(sigma, 2.0f))));
   }
   else {
      return expf(-(powf(xVal - mean, 2.0f) / (2.0f * powf((sigma / 2.0f), 2.0f))));
   }
}

BinningLayer::~BinningLayer() {}

} /* namespace PV */
