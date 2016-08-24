#pragma once

#include "utils/Image.hpp"
#include "InputLayer.hpp"

namespace PV {

   class ImageLayer : public InputLayer {

   protected:
      ImageLayer();
      int initialize(const char * name, HyPerCol * hc);
      virtual Buffer retrieveData(std::string filename, int batchIndex);
      virtual void readImage(std::string filename);
      virtual int postProcess(double timef, double dt);

   public:
      ImageLayer(const char * name, HyPerCol * hc);
      virtual ~ImageLayer();

   private:
      int initialize_base();

   protected:
      std::unique_ptr<Image> mImage = nullptr;

   };
}
