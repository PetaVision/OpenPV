#include "RotateActivityBuffer.hpp"
#include "checkpointing/CheckpointEntryRandState.hpp"
#include "utils/BufferUtilsMPI.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include <algorithm> // copy()
#include <cmath>
#include <utility>   // swap()

namespace PV {

RotateActivityBuffer::RotateActivityBuffer(
      char const *name, PVParams *params, Communicator const *comm) {
   initialize(name, params, comm);
}

void RotateActivityBuffer::initialize(
      char const *name, PVParams *params, Communicator const *comm) {
   HyPerActivityBuffer::initialize(name, params, comm);
}

int RotateActivityBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerActivityBuffer::ioParamsFillGroup(ioFlag);
   if (status != PV_SUCCESS) {
      return status;
   }

   ioParam_angleMin(ioFlag);
   ioParam_angleMax(ioFlag);
   if (ioFlag == PARAMS_IO_READ and mAngleMax < mAngleMin) {
      WarnLog().printf(
            "%s: specified value of max (%f) is less than specified value of min (%f)\n",
            getDescription_c(),
            static_cast<double>(mAngleMax),
            static_cast<double>(mAngleMin));
      std::swap(mAngleMax, mAngleMin);
      WarnLog().printf(
            "Switching angleMin to %f and angleMax to %f\n",
            static_cast<double>(mAngleMax),
            static_cast<double>(mAngleMin));
   }
   pvAssert(mAngleMax >= mAngleMin);
   ioParam_angleUnits(ioFlag);
   return status;
}

void RotateActivityBuffer::ioParam_angleMin(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValueRequired(ioFlag, getName(), "angleMin", &mAngleMin);
}

void RotateActivityBuffer::ioParam_angleMax(enum ParamsIOFlag ioFlag) {
   parameters()->ioParamValueRequired(ioFlag, getName(), "angleMax", &mAngleMax);
}

void RotateActivityBuffer::ioParam_angleUnits(enum ParamsIOFlag ioFlag) {
   if (ioFlag == PARAMS_IO_READ) {
      char *angleUnitsParam = nullptr;
      parameters()->ioParamStringRequired(ioFlag, getName(), "angleUnits", &angleUnitsParam);
      std::string angleUnitsString(angleUnitsParam);
      free(angleUnitsParam);
      for (auto &c : angleUnitsString) {
         c = tolower(c);
      }
      if (angleUnitsString == "degree") {
         angleUnitsString = "degrees";
      }
      if (angleUnitsString == "radian") {
         angleUnitsString = "radians";
      }
      if (angleUnitsString == "degrees") {
         mAngleConversionFactor = std::atan(1.0f) / 45.0f;
         mAngleUnitType = AngleUnitType::DEGREES;
      }
      else if (angleUnitsString == "radians") {
         mAngleConversionFactor = 1.0f;
         mAngleUnitType = AngleUnitType::RADIANS;
      }
   }
   else if (ioFlag == PARAMS_IO_WRITE) {
      std::string angleUnitsString;
      switch(mAngleUnitType) {
         case AngleUnitType::UNSET:
            Fatal().printf("%s AngleUnits parameter has not been set.\n", getDescription_c());
            break;
         case AngleUnitType::DEGREES:
            angleUnitsString = "degrees";
            break;
         case AngleUnitType::RADIANS:
            angleUnitsString = "radians";
            break;
         default:
            Fatal().printf("%s has an unrecognized AngleUnits parameter.\n", getDescription_c());
            break;
      }
      char *angleUnitsParam = strdup(angleUnitsString.c_str());
      FatalIf(
            angleUnitsParam == nullptr,
            "%s unable to write AngleUnitsString parameter\n",
            getDescription_c());
      parameters()->ioParamStringRequired(ioFlag, getName(), "angleUnits", &angleUnitsParam);
      free(angleUnitsParam);
   }
   else {
      Fatal().printf(
            "%s called ioParam_angleUnits with an unrecognized ioFlag.\n",
            getDescription_c());
   }
}

void RotateActivityBuffer::setObjectType() { mObjectType = "RotateActivityBuffer"; }

Response::Status RotateActivityBuffer::allocateDataStructures() {
   auto status = HyPerActivityBuffer::allocateDataStructures();
   if (!Response::completed(status)) {
      return status;
   }
   if (getCommunicator()->commRank() == 0) {
      mRandState = std::make_shared<Random>(getLayerLoc()->nbatchGlobal);
   }
   return Response::Status::SUCCESS;
}

void RotateActivityBuffer::applyTransformCPU(
      Buffer<float> const &inputBuffer, Buffer<float> &outputBuffer, float angle) {
   int const nxRes = inputBuffer.getWidth();
   int const nxExt = outputBuffer.getWidth();
   int const xMargin = (nxExt - nxRes) / 2; // assumes left and right margins are equal
   int const nyRes = inputBuffer.getHeight();
   int const nyExt = outputBuffer.getHeight();
   int const yMargin = (nyExt - nyRes) / 2; // assumes bottom and top margins are equal
   int const nf = inputBuffer.getFeatures();

   float const xCenter = 0.5f * static_cast<float>(nxRes - 1);
   float const yCenter = 0.5f * static_cast<float>(nyRes - 1);

   int const numExtended = outputBuffer.getTotalElements();
   int const nbatch = getLayerLoc()->nbatch;
   float sina = std::sin(angle);
   float cosa = std::cos(angle);
   for (int kExt = 0; kExt < numExtended; ++kExt) {
      int const kxExt = kxPos(kExt, nxExt, nyExt, nf);
      int const kyExt = kyPos(kExt, nxExt, nyExt, nf);
      int const kf    = featureIndex(kExt, nxExt, nyExt, nf);
      float xExt = static_cast<float>(kxExt - xMargin) - xCenter;
      float yExt = static_cast<float>(kyExt - yMargin) - yCenter;

      float xSrc = cosa * xExt - sina * yExt + xCenter;
      float ySrc = sina * xExt + cosa * yExt + yCenter;

      float result = interpolate(inputBuffer, xSrc, ySrc, kf);
      outputBuffer.set(kExt, result);
   }
}

float RotateActivityBuffer::interpolate(
      Buffer<float> const &inputBuffer, float xSrc, float ySrc, int feature) {
   int const nx = inputBuffer.getWidth();
   int const ny = inputBuffer.getHeight();
   int const nf = inputBuffer.getFeatures();

   float xSrcFloor = std::floor(xSrc);
   float xSrcInt   = static_cast<int>(xSrcFloor);
   float xSrcFrac  = xSrc - xSrcFloor;

   float ySrcFloor = std::floor(ySrc);
   float ySrcInt   = static_cast<int>(ySrcFloor);
   float ySrcFrac  = ySrc - ySrcFloor;
   
   float valueTL = (xSrcInt >= 0 and xSrcInt < nx and ySrcInt >= 0 and ySrcInt < ny) ?
                   inputBuffer.at(xSrcInt, ySrcInt, feature) : 0.0f;
   valueTL *= (1.0f - xSrcFrac) * (1.0f - ySrcFrac);

   float valueTR = (xSrcInt + 1 >= 0 and xSrcInt + 1 < nx and ySrcInt >= 0 and ySrcInt < ny) ?
                   inputBuffer.at(xSrcInt + 1, ySrcInt, feature) : 0.0f;
   valueTR *= xSrcFrac * (1.0f - ySrcFrac);

   float valueBL = (xSrcInt >= 0 and xSrcInt < nx and ySrcInt + 1 >= 0 and ySrcInt + 1 < ny) ?
                   inputBuffer.at(xSrcInt, ySrcInt + 1, feature) : 0.0f;
   valueBL *= (1.0f - xSrcFrac) * ySrcFrac;

   float valueBR = (xSrcInt + 1 >= 0 and xSrcInt + 1 < nx and ySrcInt + 1 >= 0 and ySrcInt + 1 < ny) ?
                   inputBuffer.at(xSrcInt + 1, ySrcInt + 1, feature) : 0.0f;
   valueBR *= xSrcFrac * ySrcFrac;

   float value = valueTL + valueTR + valueBL + valueBR;
   return value;
}

void RotateActivityBuffer::updateBufferCPU(double simTime, double deltaTime) {
   auto localMPIBlock = getCommunicator()->getLocalMPIBlock();
   PVLayerLoc const *loc = getLayerLoc();
   Buffer<float> localVBuffer(loc->nx, loc->ny, loc->nf);  
   for (int b = 0; b < loc->nbatch; ++b) {
      float const* batchStartV = mInternalState->getBufferData(b);
      auto &localVBufferVector = localVBuffer.asVector();
      std::copy(batchStartV, &batchStartV[mBufferSize], &localVBufferVector[0]);
      Buffer<float> globalV = BufferUtils::gather(
            localMPIBlock,
            localVBuffer,
            loc->nx,
            loc->ny,
            0 /* Local communicator always has batch dimension of 1 */,
            0 /* destination process */);
      // root process of local communicator now has entire batch element in globalV buffer.
      Buffer<float> activityBuffer;
      if (localMPIBlock->getRank() == 0) {
         // Set activityBuffer buffer to size of global extended buffer, to hold result of rotation
         int const nxGlobalExt = loc->nxGlobal + loc->halo.lt + loc->halo.rt;
         int const nyGlobalExt = loc->nyGlobal + loc->halo.dn + loc->halo.up;
         activityBuffer.resize(nxGlobalExt, nyGlobalExt, loc->nf);
         int globalMPIBatchIndex = localMPIBlock->getStartBatch() + localMPIBlock->getBatchIndex();
         int bGlobal = b + loc->nbatch * globalMPIBatchIndex;
         float angle = mRandState->uniformRandom(bGlobal, mAngleMin, mAngleMax);
         float angleRadians = mAngleConversionFactor * angle;
         applyTransformCPU(globalV, activityBuffer, angleRadians);
      }
      else {
         int const nxLocalExt = loc->nx + loc->halo.lt + loc->halo.rt;
         int const nyLocalExt = loc->ny + loc->halo.dn + loc->halo.up;
         activityBuffer.resize(nxLocalExt, nyLocalExt, loc->nf);
      }
      BufferUtils::scatter(localMPIBlock, activityBuffer, loc->nx, loc->ny, 0, 0);
      auto const &activityBufferVector = activityBuffer.asVector();
      int const numLocalExtended = activityBuffer.getTotalElements();
      std::copy(
            &activityBufferVector[0],
            &activityBufferVector[numLocalExtended],
            &mBufferData[b * mBufferSize]);
   }
}

} // namespace PV
