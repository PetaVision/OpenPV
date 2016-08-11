/*
 * ImageLayer.hpp
 *
 *    Layer that represents an image or list of images
 *    loaded from disk. Supports .jpg, .png, and .bmp
 *    formats, or a .txt list of files in those formats.
 */


#ifndef IMAGELAYER_HPP_
#define IMAGELAYER_HPP_

#include "utils/Image.hpp"
#include "InputLayer.hpp"

#include <cMakeHeader.h>

namespace PV {

   class ImageLayer : public InputLayer {

   protected:
      ImageLayer();
      int initialize(const char * name, HyPerCol * hc);
      virtual Buffer retrieveData(std::string filename);
      virtual void readImage(std::string filename);
      virtual int postProcess(double timef, double dt);
      virtual bool readyForNextFile();

   public:
      ImageLayer(const char * name, HyPerCol * hc);
      virtual ~ImageLayer();

   private:
      int initialize_base();

   protected:
      std::unique_ptr<Image> mImage;

   };
}

#endif 
