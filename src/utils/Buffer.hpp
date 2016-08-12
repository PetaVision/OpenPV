#pragma once

#include <vector>
#include <cmath>

namespace PV {

   class Buffer {
      public:
         typedef std::vector< std::vector< std::vector<float> > > Vec3;
         
         enum RescaleMethod {
            CROP,
            PAD
         };
         
         enum InterpolationMethod {
            NEAREST,
            BICUBIC
         };

         enum OffsetAnchor {
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

         enum PointConstraintMethod {
            CLAMP,
            MIRROR,
            WRAP
         };

         Buffer(int rows, int columns, int features); 
         Buffer();
         // TODO: Add a constructor that calls resize() and then set(vector)
         // TODO: Use that constructor to implement a copy constructor as well
         float at(int x, int y, int feature); 
         void set(int x, int y, int feature, float value);
         void set(const std::vector<float> &vector);
         void resize(int width, int height, int features);
         void crop(int targetWidth, int targetHeight, enum OffsetAnchor offsetAnchor, int offsetX, int offsetY);
         void rescale(int targetWidth, int targetHeight, enum RescaleMethod rescaleMethod, enum InterpolationMethod interpMethod);
         std::vector<float> asVector();

         int getHeight()   { return mData.size(); }
         int getWidth()    { return mData.at(0).size(); }
         int getFeatures() { return mData.at(0).at(0).size(); }

         // TODO: Can these be protected?
         static int getOffsetX(enum OffsetAnchor offsetAnchor, int offsetX, int newWidth, int currentWidth);
         static int getOffsetY(enum OffsetAnchor offsetAnchor, int offsetY, int newHeight, int currentHeight);
         static bool constrainPoint(int &x, int &y, int minX, int maxX, int minY, int maxY, enum PointConstraintMethod method);

      protected:
         // TODO: Data should be stored as a one dimensional vector instead of this. Do indexing math in at / set methods.
         Vec3 mData;

      private:
         static void nearestNeighborInterp(float const * bufferIn, int widthIn, int heightIn, int numBands, int xStrideIn, int yStrideIn, int bandStrideIn, float * bufferOut, int widthOut, int heightOut);
         static void bicubicInterp(float const * bufferIn, int widthIn, int heightIn, int numBands, int xStrideIn, int yStrideIn, int bandStrideIn, float * bufferOut, int widthOut, int heightOut);
         inline static float bicubic(float x) {
            float const absx = fabsf(x);
            return absx < 1 ? 1 + absx*absx*(-2 + absx) : absx < 2 ? 4 + absx*(-8 + absx*(5-absx)) : 0;
         }



   };

}
