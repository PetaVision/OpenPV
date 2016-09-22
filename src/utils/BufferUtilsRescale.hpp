#ifndef __BUFFERUTILSRESCALE_HPP__
#define __BUFFERUTILSRESCALE_HPP__

#include "structures/Buffer.hpp"

namespace PV {
   namespace BufferUtils {

      enum InterpolationMethod {
         NEAREST,
         BICUBIC
      };

      enum RescaleMethod {
         CROP,
         PAD
      };

      void rescale(Buffer<float> &buffer,
                   int newWidth,
                   int newHeight,
                   enum RescaleMethod rescaleMethod,
                   enum InterpolationMethod interpMethod,
                   enum Buffer<float>::Anchor anchor);

   } // End BufferUtils namespace
} // End PV namespace
#endif




