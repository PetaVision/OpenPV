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

         Buffer(int rows, int columns, int features);
         Buffer();
         float at(int row, int column, int feature); 
         void set(int row, int column, int feature, float value);
         void set(const std::vector<float> &vector);
         void resize(int rows, int columns, int features);
         void rescale(int targetRows, int targetColumns, enum RescaleMethod rescaleMethod, enum InterpolationMethod interpMethod);
         std::vector<float> asVector();

         int getRows() { return mData.size(); }
         int getColumns() { return mData.at(0).size(); }
         int getFeatures() { return mData.at(0).at(0).size(); }
         
         static void nearestNeighborInterp(float const * bufferIn, int widthIn, int heightIn, int numBands, int xStrideIn, int yStrideIn, int bandStrideIn, float * bufferOut, int widthOut, int heightOut);
         static void bicubicInterp(float const * bufferIn, int widthIn, int heightIn, int numBands, int xStrideIn, int yStrideIn, int bandStrideIn, float * bufferOut, int widthOut, int heightOut);

      protected:
         Vec3 mData;
         

      inline static float bicubic(float x) {
            float const absx = fabsf(x); // assumes float is float ; ideally should generalize
            return absx < 1 ? 1 + absx*absx*(-2 + absx) : absx < 2 ? 4 + absx*(-8 + absx*(5-absx)) : 0;

         }



   };

}
