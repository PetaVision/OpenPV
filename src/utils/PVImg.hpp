// Wrapper class for stb_image.h
// athresher, Jul 27 2016

#ifndef __PVIMG_HPP__
#define __PVIMG_HPP__

# ifndef STB_IMAGE_IMPLEMENTATION
#  define STB_IMAGE_IMPLEMENTATION 
#  include "stb_image.h"
# endif

#include <string>
#include <vector>

namespace PV {

   class PVImg {

      public:
         PVImg(std::string filename);
         PVImg(const std::vector<float> &data, int width, int height, int channels);

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
