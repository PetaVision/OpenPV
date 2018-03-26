/*
 * ImageFromMemoryBuffer.hpp
 *
 *  Created on: Oct 31, 2014
 *      Author: Pete Schultz
 *  Description of the class is in ImageFromMemoryBuffer.hpp
 */

#include "ImageFromMemoryBuffer.hpp"

#include <vector>

namespace PV {

ImageFromMemoryBuffer::ImageFromMemoryBuffer(char const *name, HyPerCol *hc) {
   initialize_base();
   initialize(name, hc);
}

ImageFromMemoryBuffer::ImageFromMemoryBuffer() { initialize_base(); }

int ImageFromMemoryBuffer::initialize_base() {
   hasNewImageFlag = false;
   mAutoResizeFlag = false;
   return PV_SUCCESS;
}

int ImageFromMemoryBuffer::initialize(char const *name, HyPerCol *hc) {
   if (mUseInputBCflag && mAutoResizeFlag) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "%s: setting both useInputBCflag and autoResizeFlag has not yet been implemented.\n",
               getDescription_c());
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}

int ImageFromMemoryBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ImageLayer::ioParamsFillGroup(ioFlag);
   ioParam_autoResizeFlag(ioFlag);
   ioParam_aspectRatioAdjustment(ioFlag);
   return status;
}

template <typename pixeltype>
int ImageFromMemoryBuffer::setMemoryBuffer(
      pixeltype const *externalBuffer,
      int height,
      int width,
      int numbands,
      int xstride,
      int ystride,
      int bandstride,
      pixeltype zeroval,
      pixeltype oneval) {
   if (height <= 0 || width <= 0 || numbands <= 0) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "ImageFromMemoryBuffer::setMemoryBuffer: height, width, numbands "
               "arguments must be positive.\n");
      }
      return PV_FAILURE;
   }

   int newSize = height * width * numbands;

   if (parent->columnId() == 0) {
      std::vector<float> newData(newSize);
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for (int k = 0; k < newSize; k++) {
         int x             = kxPos(k, width, height, numbands);
         int y             = kyPos(k, width, height, numbands);
         int f             = featureIndex(k, width, height, numbands);
         int externalIndex = x * xstride + y * ystride + f * bandstride;
         newData.at(k)     = pixelTypeConvert(externalBuffer[externalIndex], zeroval, oneval);
      }
      mImage = std::unique_ptr<Image>(new Image(newData, width, height, numbands));
   }
   hasNewImageFlag = true;

   return PV_SUCCESS;
}
template int ImageFromMemoryBuffer::setMemoryBuffer<uint8_t>(
      uint8_t const *buffer,
      int height,
      int width,
      int numbands,
      int xstride,
      int ystride,
      int bandstride,
      uint8_t zeroval,
      uint8_t oneval);

template <typename pixeltype>
int ImageFromMemoryBuffer::setMemoryBuffer(
      pixeltype const *externalBuffer,
      int height,
      int width,
      int numbands,
      int xstride,
      int ystride,
      int bandstride,
      pixeltype zeroval,
      pixeltype oneval,
      int offsetX,
      int offsetY,
      char const *offsetAnchor) {
   mOffsetX = offsetX;
   mOffsetY = offsetY;
   mAnchor  = Buffer<float>::CENTER;
   if (checkValidAnchorString(offsetAnchor) != PV_SUCCESS) {
      if (parent->columnId() == 0) {
         ErrorLog().printf(
               "%s: setMemoryBuffer called with invalid anchor string \"%s\"",
               getDescription_c(),
               offsetAnchor);
      }
      MPI_Barrier(parent->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   // TODO: This is a dirty hack, make this function take an enum instead of a string to fix it
   if (strcmp(offsetAnchor, "cc") != 0) {
      WarnLog() << getName() << ": Offset anchor %s is being ignored, using cc instead. "
                                "ImageFromMemoryBuffer only supports cc.\n";
   }
   return setMemoryBuffer(
         externalBuffer, height, width, numbands, xstride, ystride, bandstride, zeroval, oneval);
}
template int ImageFromMemoryBuffer::setMemoryBuffer<uint8_t>(
      uint8_t const *buffer,
      int height,
      int width,
      int numbands,
      int xstride,
      int ystride,
      int bandstride,
      uint8_t zeroval,
      uint8_t oneval,
      int offsetX,
      int offsetY,
      char const *offsetAnchor);

template <typename pixeltype>
float ImageFromMemoryBuffer::pixelTypeConvert(pixeltype q, pixeltype zeroval, pixeltype oneval) {
   return ((float)(q - zeroval)) / ((float)(oneval - zeroval));
}
template float ImageFromMemoryBuffer::pixelTypeConvert<unsigned char>(
      unsigned char q,
      unsigned char zeroval,
      unsigned char oneval);

void ImageFromMemoryBuffer::initializeActivity(double time, double dt) { retrieveInput(time, dt); }

Response::Status ImageFromMemoryBuffer::updateState(double time, double dt) {
   assert(hasNewImageFlag); // updateState shouldn't have been called otherwise.
   Fatal() << "ImageFromMemoryBuffer is currently broken.\n"; // Marked broken Apr 24, 2017.
   hasNewImageFlag = false;
   // TODO: Need to refactor ImageFromMemoryBuffer to reflect the refactoring of the rest of
   // the InputLayer hierarchy.
   return Response::SUCCESS;
}

double ImageFromMemoryBuffer::getDeltaUpdateTime() { return parent->getStopTime(); }

ImageFromMemoryBuffer::~ImageFromMemoryBuffer() {}

} // namespace PV
