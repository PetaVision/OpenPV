/*
 * ImageFromMemoryBuffer.hpp
 *
 *  Created on: Oct 31, 2014
 *      Author: Pete Schultz
 *  Description of the class is in ImageFromMemoryBuffer.hpp
 */

#include "ImageFromMemoryBuffer.hpp"

namespace PV {

ImageFromMemoryBuffer::ImageFromMemoryBuffer(char const * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

ImageFromMemoryBuffer::ImageFromMemoryBuffer() {
   initialize_base();
   // protected default constructor; initialize(name,hc) should be called by any derived class's initialization routine
}

int ImageFromMemoryBuffer::initialize_base() {
   hasNewImageFlag = false;
   autoResizeFlag = false;
   aspectRatioAdjustment = NULL;
   resizeFactor = 1.0f;
   return PV_SUCCESS;
}

int ImageFromMemoryBuffer::initialize(char const * name, HyPerCol * hc) {
   return BaseInput::initialize(name, hc);
   if (useImageBCflag && autoResizeFlag) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: setting both useImageBCflag and autoResizeFlag has not yet been implemented.\n", getKeyword(), name);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
}

int ImageFromMemoryBuffer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseInput::ioParamsFillGroup(ioFlag);
   ioParam_autoResizeFlag(ioFlag);
   ioParam_aspectRatioAdjustment(ioFlag);
   return status;
}

template <typename pixeltype>
int ImageFromMemoryBuffer::setMemoryBuffer(pixeltype const * externalBuffer, int height, int width, int numbands, int xstride, int ystride, int bandstride, pixeltype zeroval, pixeltype oneval) {
   if (height<=0 || width<=0 || numbands<=0) {
      if (parent->columnId()==0) {
         fprintf(stderr, "ImageFromMemoryBuffer::setMemoryBuffer error: height, width, numbands arguments must be positive.\n");
      }
      return PV_FAILURE;
   }

   int oldSize = imageLoc.nxGlobal * imageLoc.nyGlobal * imageLoc.nf;
   int newSize = height * width * numbands;

   if (oldSize != newSize) {
      delete[] imageData;
      imageData = new pvadata_t[newSize];
      imageLoc.nxGlobal = width;
      imageLoc.nyGlobal = height;
      imageLoc.nf = numbands;
   }

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif // PV_USE_OPENMP_THREADS
   for (int k=0; k<newSize; k++) {
      int x=kxPos(k, imageLoc.nxGlobal, imageLoc.nyGlobal, imageLoc.nf);
      int y=kyPos(k, imageLoc.nxGlobal, imageLoc.nyGlobal, imageLoc.nf);
      int f=featureIndex(k, imageLoc.nxGlobal, imageLoc.nyGlobal, imageLoc.nf);
      int externalIndex = x*xstride + y*ystride + f*bandstride;
      imageData[k] = pixelTypeConvert(externalBuffer[externalIndex], zeroval, oneval);
   }

   switch (numbands) {
   case 1:
   case 2: // fall-through is deliberate
      imageColorType = COLORTYPE_GRAYSCALE;
      break;
   case 3:
   case 4: // fall-through is deliberate
      imageColorType = COLORTYPE_RGB; // Only supporting RGB for now.  TODO: add YUV etc.
      break;
   default:
      imageColorType = COLORTYPE_UNRECOGNIZED;
      break;
   }
   hasNewImageFlag = true;

   return PV_SUCCESS;
}
template int ImageFromMemoryBuffer::setMemoryBuffer<uint8_t>(uint8_t const * buffer, int height, int width, int numbands, int xstride, int ystride, int bandstride, uint8_t zeroval, uint8_t oneval);

template <typename pixeltype>
int ImageFromMemoryBuffer::setMemoryBuffer(pixeltype const * externalBuffer, int height, int width, int numbands, int xstride, int ystride, int bandstride, pixeltype zeroval, pixeltype oneval, int offsetX, int offsetY, char const * offsetAnchor) {
   offsets[0] = offsetX;
   offsets[1] = offsetY;
   free(this->offsetAnchor);
   this->offsetAnchor = strdup(offsetAnchor);
   if (checkValidAnchorString()!=PV_SUCCESS) {
      if (parent->columnId()==0) {
         fprintf(stderr, "%s \"%s\" error: setMemoryBuffer called with invalid anchor string \"%s\"",
               getKeyword(), name, offsetAnchor);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   return setMemoryBuffer(externalBuffer, height, width, numbands, xstride, ystride, bandstride, zeroval, oneval);
}
template int ImageFromMemoryBuffer::setMemoryBuffer<uint8_t>(uint8_t const * buffer, int height, int width, int numbands, int xstride, int ystride, int bandstride, uint8_t zeroval, uint8_t oneval, int offsetX, int offsetY, char const * offsetAnchor);

template <typename pixeltype>
pvadata_t ImageFromMemoryBuffer::pixelTypeConvert(pixeltype q, pixeltype zeroval, pixeltype oneval) {
   return ((pvadata_t) (q-zeroval))/((pvadata_t) (oneval-zeroval));
}
template pvadata_t ImageFromMemoryBuffer::pixelTypeConvert<unsigned char>(unsigned char q, unsigned char zeroval, unsigned char oneval);

int ImageFromMemoryBuffer::initializeActivity(double time, double dt) {
   return getFrame(time, dt);
}

int ImageFromMemoryBuffer::updateState(double time, double dt) {
   assert(hasNewImageFlag); // updateState shouldn't have been called otherwise.
   hasNewImageFlag = false;
   return getFrame(time, dt);
}

int ImageFromMemoryBuffer::retrieveData(double timef, double dt, int batchIndex)
{
   pvAssert(batchIndex==0); // ImageFromMemoryBuffer is not batched.
   return PV_SUCCESS; // imageData, imageLoc, imageColorType were already set in setMemoryBuffer
}

double ImageFromMemoryBuffer::getDeltaUpdateTime(){
   return parent->getStopTime() - parent->getStartTime();
}

int ImageFromMemoryBuffer::outputState(double time, bool last) {
   return HyPerLayer::outputState(time, last);
}


ImageFromMemoryBuffer::~ImageFromMemoryBuffer() {
}

BaseObject * createImageFromMemoryBuffer(char const * name, HyPerCol * hc) {
   return hc ? new ImageFromMemoryBuffer(name, hc) : NULL;
}

}  // namespace PV
