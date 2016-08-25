#pragma once

#include "utils/Image.hpp"
#include "InputLayer.hpp"

namespace PV {

   class ImageLayer : public InputLayer {

   protected:
      ImageLayer() {}
      virtual Buffer retrieveData(std::string filename, int batchIndex);
      void readImage(std::string filename);

   public:
      ImageLayer(const char * name, HyPerCol * hc);
      virtual ~ImageLayer() {}

   protected:
      std::unique_ptr<Image> mImage = nullptr;

   };
}
