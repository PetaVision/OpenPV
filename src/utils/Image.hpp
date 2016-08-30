// Wrapper class for stb_image.h
// athresher, Jul 27 2016

#pragma once

#include "Buffer.hpp"

#include <string>
#include <vector>

namespace PV {
   
   class Image : public Buffer { 
      public:
         Image(std::string filename);
         Image(const std::vector<float> &data, int width, int height, int channels);

         void setPixel(int x, int y, float r, float g, float b);
         void setPixel(int x, int y, float r, float g, float b, float a);
         float getPixelR(int x, int y);
         float getPixelG(int x, int y);
         float getPixelB(int x, int y);
         float getPixelA(int x, int y);
         void convertToColor(bool alphaChannel);
         void convertToGray(bool alphaChannel);
         void read(std::string filename);
         void write(std::string filename); 
         static constexpr const float mRToGray = 0.30f;
         static constexpr const float mGToGray = 0.59f;
         static constexpr const float mBToGray = 0.11f;

      protected:
         // These only line up for RGB and RGBA. Should that change?
         const int mRPos = 0;
         const int mGPos = 1;
         const int mBPos = 2;
         const int mAPos = 3;
   };

}
