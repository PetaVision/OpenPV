#ifndef IMAGELAYER_HPP_
#define IMAGELAYER_HPP_

#include "InputLayer.hpp"
#include "structures/Image.hpp"

namespace PV {

class ImageLayer : public InputLayer {

  protected:
   ImageLayer() {}
   virtual int countInputImages() override;
   void populateFileList();
   virtual Buffer<float> retrieveData(int inputIndex) override;
   virtual std::string describeInput(int index) override;
   void readImage(std::string filename);

  public:
   ImageLayer(const char *name, HyPerCol *hc);
   virtual ~ImageLayer() {}
   virtual std::string const &
   getCurrentFilename(int localBatchElement, int mpiBatchIndex) const override;

  protected:
   std::unique_ptr<Image> mImage = nullptr;

   // Automatically set if the inputPath ends in .txt. Determines whether this layer represents a
   // collection of files.
   bool mUsingFileList = false;

   // List of filenames to iterate over
   std::vector<std::string> mFileList;
}; // end class ImageLayer
} // end namespace PV

#endif // IMAGELAYER_HPP_
