#ifndef __REALBUFFER_HPP__
#define __REALBUFFER_HPP__

#include "Buffer.hpp"

#include <cmath>

namespace PV {

class RealBuffer : public Buffer<float> {

   public:
      using Buffer<float>::Buffer;

      enum InterpolationMethod {
         NEAREST,
         BICUBIC
      };

      enum RescaleMethod {
         CROP,
         PAD
      };
 
      void rescale(int newWidth, int newHeight,
            enum RescaleMethod rescaleMethod,
            enum InterpolationMethod interpMethod,
            enum Anchor anchor);
 
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
};

}
#endif






