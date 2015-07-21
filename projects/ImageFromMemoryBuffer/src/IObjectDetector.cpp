/*
 * IObjectDetector
 *
 * Created on Nov 3, 2014
 *     Author: Pete Schultz
 *
 * Description of the class is in IObjectDetector.hpp
 */

#include "IObjectDetector.hpp"

namespace vidint {

int32_t IObjectDetector::detect(Image* image, Roi* roi, Object* results, uint32_t max, uint32_t* resultCount) {
   int height = image->height;
   int width = image->width;
   int numbands = -1;
   switch(image->type) {
      case GRAY8: numbands = 1; break;
      case RGB24: numbands = 3; break;
      case YV12: fprintf(stderr, "YV12 not implemented yet :(\n"); exit(1); break;
      default:   assert(0); /*Nothing other than GRAY8, RGB24, YV12 should be possible*/ break;
   }
   assert(numbands > 0);
   int xstride = numbands;
   int ystride = numbands*image->width;
   int bandstride = 1;
   uint8_t zeroval = (uint8_t) 0;
   uint8_t oneval = (uint8_t) 255;
   // NB: make sure choice of imagestridex, imagestridey, imagestridebands is correct
   const int imagestridex = numbands;
   const int imagestridey = numbands*image->width;
   const int imagestridebands = 1;
   uint8_t * pPixel = image==NULL ? NULL : image->pPixel; // Under MPI, only the root process needs to receive the image pixels
   imagelayer->setMemoryBuffer<uint8_t>(pPixel, height, width, numbands, xstride, ystride, bandstride, zeroval, oneval, roi->x, roi->y, "tl");
   double newStartTime = hypercolumn->simulationTime();
   double newStopTime = newStartTime + timeInterval;
   hypercolumn->run(newStartTime, newStopTime, hypercolumn->getDeltaTime());
   *resultCount = 0;
   return 0;
}

IObjectDetector::IObjectDetector(PV::HyPerCol * hc, char const * imagelayername, double simTimeLength) {
   initialize_base();
   initialize(hc, imagelayername, simTimeLength);
}

IObjectDetector::IObjectDetector() {
   initialize_base();
}

int IObjectDetector::initialize_base() {
   hypercolumn = NULL;
   return VIDINT_SUCCESS;
}

int IObjectDetector::initialize(PV::HyPerCol * hc, char const * imagelayername, double simTimeInterval) {
   this->hypercolumn = hc;
   PV::HyPerLayer * layer = hc->getLayerFromName(imagelayername);
   if (layer==NULL) {
      if (hc->columnId()==0) {
         fprintf(stderr, "IObjectDetector error: \"%s\" is not a layer in the HyPerColumn \"%s\".\n",
               imagelayername, hc->getName());
      }
      return VIDINT_FAILURE;
   }
   this->imagelayer = dynamic_cast<PV::ImageFromMemoryBuffer *>(layer);
   if (this->imagelayer==NULL) {
      if (hc->columnId()==0) {
         fprintf(stderr, "IObjectDetector error: %s \"%s\" must be an ImageFromMemoryBuffer layer.\n",
               hc->parameters()->groupKeywordFromName(imagelayername), imagelayername);
      }
      return VIDINT_FAILURE;
   }
   this->timeInterval = simTimeInterval;
   return VIDINT_SUCCESS;
}

IObjectDetector::~IObjectDetector() {
}

}  // namespace vidint