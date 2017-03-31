#pragma once

#include "InputLayer.hpp"
#include "structures/Image.hpp"

namespace PV {

class ImageLayer : public InputLayer {

  protected:
   ImageLayer() {}
   virtual int countInputImages() override;
   virtual Buffer<float> retrieveData(std::string filename, int batchIndex) override;
   void readImage(std::string filename);

  public:
   ImageLayer(const char *name, HyPerCol *hc);
   virtual ~ImageLayer() {}

  protected:
   std::unique_ptr<Image> mImage = nullptr;
};
}
