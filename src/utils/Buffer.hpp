#pragma once

#include <vector>
#include <cmath>

namespace PV {

   class Buffer {
      public:
         enum RescaleMethod {
            CROP,
            PAD
         };
         
         enum InterpolationMethod {
            NEAREST,
            BICUBIC
         };

         enum Anchor {
            CENTER,
            NORTH,
            NORTHEAST,
            EAST,
            SOUTHEAST,
            SOUTH,
            SOUTHWEST,
            WEST,
            NORTHWEST
         };

         Buffer(int width, int height, int features); 
         Buffer();
         Buffer(const std::vector<float> &data, int width, int height, int features);

         float at(int x, int y, int feature); 
         void set(int x, int y, int feature, float value);
         void set(const std::vector<float> &vector, int width, int height, int features);
         void resize(int width, int height, int features);
         void crop(int newWidth, int newHeight, enum Anchor anchor);
         void grow(int newWidth, int newHeight, enum Anchor anchor);
         void rescale(int newWidth, int newHeight,
               enum RescaleMethod rescaleMethod,
               enum InterpolationMethod interpMethod,
               enum Anchor anchor);
         void translate(int offsetX, int offsetY);
         std::vector<float> asVector() { return mData; }
         int getHeight()   { return mHeight; }
         int getWidth()    { return mWidth; }
         int getFeatures() { return mFeatures; }

      protected:
         static int getAnchorX(enum Anchor anchor, int smallerWidth, int biggerWidth);
         static int getAnchorY(enum Anchor anchor, int smallerHeight, int biggerHeight);

      private:
         static void nearestNeighborInterp(float const *bufferIn,
               int widthIn,   int heightIn,  int numBands,
               int xStrideIn, int yStrideIn, int bandStrideIn,
               float *bufferOut,
               int widthOut,  int heightOut);
         static void bicubicInterp(float const *bufferIn,
               int widthIn,   int heightIn,  int numBands,
               int xStrideIn, int yStrideIn, int bandStrideIn,
               float *bufferOut,
               int widthOut,  int heightOut);

         inline static float bicubic(float x) {
            float const absx = fabsf(x);
            return  absx < 1 ? 1 + absx * absx * (-2 + absx)
                  : absx < 2 ? 4 + absx * (-8 + absx * (5 - absx))
                  : 0;
         }

         inline int index(int x, int y, int f) {
            return f + (x + y * mWidth) * mFeatures;
         }
         
         std::vector<float> mData;
         int mWidth    = 0;
         int mHeight   = 0;
         int mFeatures = 0;      
   };
}
