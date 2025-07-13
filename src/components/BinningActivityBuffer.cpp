/*
 * BinningActivityBuffer.cpp
 *
 *  Created on: Jan 15, 2014
 *      Author: Sheng Lundquist
 */

#include "BinningActivityBuffer.hpp"
#include "components/OriginalLayerNameParam.hpp"

namespace PV {

BinningActivityBuffer::BinningActivityBuffer(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   initialize(params, defaults, comm);
}

BinningActivityBuffer::~BinningActivityBuffer() {}

void BinningActivityBuffer::initialize(
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) {
   ActivityBuffer::initialize(params, defaults, comm);
}

void BinningActivityBuffer::setObjectType() { mObjectType = "BinningActivityBuffer"; }

int BinningActivityBuffer::ioParamsFillGroup(ParamsIOSwitch ioSwitch) {
   int status = ActivityBuffer::ioParamsFillGroup(ioSwitch);
   ioParam_binMin(ioSwitch);
   ioParam_binMax(ioSwitch);
   ioParam_delay(ioSwitch);
   ioParam_binSigma(ioSwitch);
   ioParam_zeroNeg(ioSwitch);
   ioParam_zeroDCR(ioSwitch);
   ioParam_normalDist(ioSwitch);
   return status;
}

void BinningActivityBuffer::ioParam_binMin(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "binMin", &mBinMin);
}

void BinningActivityBuffer::ioParam_binMax(ParamsIOSwitch ioSwitch) {
   pvAssert(!mParamsIO->presentAndNotBeenRead("binMin"));
   mParamsIO->ioParam(ioSwitch, "binMax", &mBinMax);
   if (ioSwitch == ParamsIOSwitch::Read && mBinMax <= mBinMin) {
      if (mCommunicator->commRank() == 0) {
         ErrorLog().printf(
               "%s: binMax (%f) must be greater than binMin (%f).\n",
               getDescription_c(),
               (double)mBinMax,
               (double)mBinMin);
      }
      MPI_Barrier(mCommunicator->communicator());
      exit(EXIT_FAILURE);
   }
}

void BinningActivityBuffer::ioParam_binSigma(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "binSigma", &mBinSigma);
}

void BinningActivityBuffer::ioParam_delay(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "delay", &mDelay);
}

void BinningActivityBuffer::ioParam_zeroNeg(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "zeroNeg", &mZeroNeg);
}

void BinningActivityBuffer::ioParam_zeroDCR(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "zeroDCR", &mZeroDCR);
}

void BinningActivityBuffer::ioParam_normalDist(ParamsIOSwitch ioSwitch) {
   mParamsIO->ioParam(ioSwitch, "normalDist", &mNormalDist);
}

// TODO read params for gaussian over features

Response::Status BinningActivityBuffer::communicateInitInfo(
      std::shared_ptr<CommunicateInitInfoMessage const> message) {
   auto status = ActivityBuffer::communicateInitInfo(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *objectTable            = message->mObjectTable;
   auto *originalLayerNameParam = objectTable->findObject<OriginalLayerNameParam>(getName());
   FatalIf(
         originalLayerNameParam == nullptr,
         "%s could not find an OriginalLayerNameParam.\n",
         getDescription_c());
   if (!originalLayerNameParam->getInitInfoCommunicatedFlag()) {
      return Response::POSTPONE;
   }
   std::string const &originalLayerName = originalLayerNameParam->getLinkedObjectName();

   mOriginalLayerData = objectTable->findObject<BasePublisherComponent>(originalLayerName);
   FatalIf(
         mOriginalLayerData == nullptr,
         "%s original layer \"%s\" does not have a BasePublisherComponent.\n",
         getDescription_c(),
         originalLayerName.c_str());
   checkDimensions();

   mOriginalLayerData->increaseDelayLevels(mDelay);
   return Response::SUCCESS;
}

void BinningActivityBuffer::checkDimensions() const {
   PVLayerLoc const *locOriginal = mOriginalLayerData->getLayerLoc();
   PVLayerLoc const *loc         = getLayerLoc();
   FatalIf(
         locOriginal->nbatch != loc->nbatch,
         "%s and %s do not have the same batch width (%d versus %d)\n",
         mOriginalLayerData->getDescription_c(),
         getDescription_c(),
         locOriginal->nbatch,
         loc->nbatch);

   bool dimsEqual = true;
   dimsEqual      = dimsEqual and (locOriginal->nx == loc->nx);
   dimsEqual      = dimsEqual and (locOriginal->ny == loc->ny);
   FatalIf(
         !dimsEqual,
         "%s and %s do not have the same x- and y- dimensions (%d-by-%d) versus (%d-by-%d).\n",
         mOriginalLayerData->getDescription_c(),
         getDescription_c(),
         locOriginal->nx,
         locOriginal->nx,
         loc->nx,
         loc->ny);

   FatalIf(
         locOriginal->nf != 1,
         "\"%s\" requires original layer \"%s\" to have only one feature (nf=%d)\n",
         getDescription_c(),
         mOriginalLayerData->getName(),
         locOriginal->nf);
}

Response::Status
BinningActivityBuffer::initializeState(std::shared_ptr<InitializeStateMessage const> message) {
   return Response::NO_ACTION;
}

void BinningActivityBuffer::updateBufferCPU(double simTime, double deltaTime) {
   PVLayerLoc const *origLoc = mOriginalLayerData->getLayerLoc();
   PVLayerLoc const *currLoc = getLayerLoc();

   pvAssert(origLoc->nx == currLoc->nx);
   pvAssert(origLoc->ny == currLoc->ny);
   pvAssert(origLoc->nf == 1);

   pvAssert(origLoc->halo.lt == currLoc->halo.lt);
   pvAssert(origLoc->halo.rt == currLoc->halo.rt);
   pvAssert(origLoc->halo.dn == currLoc->halo.dn);
   pvAssert(origLoc->halo.up == currLoc->halo.up);

   int numBins    = currLoc->nf;
   float binRange = mBinMax - mBinMin;
   float stepSize = float(binRange) / numBins;

   int const nxExt = origLoc->nx + origLoc->halo.lt + origLoc->halo.rt;
   int const nyExt = origLoc->ny + origLoc->halo.dn + origLoc->halo.up;

   float const *origData = mOriginalLayerData->getLayerData(mDelay);
   float *currActivity   = mBufferData.data();

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
}

float BinningActivityBuffer::calcGaussian(float x, float sigma) {
   return std::exp(-x * x / (2 * sigma * sigma));
}

} // namespace PV
