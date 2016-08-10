// Wrapper class for stb_image.h
// athresher, Jul 27 2016

#ifndef IMAGE_HPP__
#define IMAGE_HPP__

#include <string>
#include <vector>

namespace PV {

   class Image { // TODO: This should probably be a subclass of Buffer

      public:
         Image(std::string filename);
         Image(const std::vector<float> &data, int width, int height, int channels);

         void setPixel(int x, int y, float r, float g, float b);
         void setPixel(int x, int y, float r, float g, float b, float a);
         void deserialize(const std::vector<float> &data, int width, int height, int channels);
         std::vector<float> serialize(int channels);
         int getWidth() { return mWidth; }
         int getHeight() { return mHeight; }
         float getPixelR(int x, int y) { return mData.at(y).at(x).at(mRPos); }
         float getPixelG(int x, int y) { return mData.at(y).at(x).at(mGPos); }
         float getPixelB(int x, int y) { return mData.at(y).at(x).at(mBPos); }
         float getPixelA(int x, int y) { return mData.at(y).at(x).at(mAPos); }

      protected:
         int mWidth = 0;
         int mHeight = 0;
         std::vector< std::vector< std::vector<float> > > mData;

         const int mRPos = 0;
         const int mGPos = 1;
         const int mBPos = 2;
         const int mAPos = 3;
   };
}

#endif
