#include "BufferUtilsRescale.hpp"
#include "conversions.h"
#include <cmath>
#include <cstring>

namespace PV {
namespace BufferUtils {
namespace { // Anonymous namespace for "private" functions

inline static float bicubic(float x) {
   float const absx = fabsf(x);
   return absx < 1 ? 1 + absx * absx * (-2 + absx)
                   : absx < 2 ? 4 + absx * (-8 + absx * (5 - absx)) : 0;
}

void nearestNeighborInterp(
      float const *bufferIn,
      int widthIn,
      int heightIn,
      int numBands,
      int xStrideIn,
      int yStrideIn,
      int bandStrideIn,
      float *bufferOut,
      int widthOut,
      int heightOut) {

   /* Interpolation using nearest neighbor interpolation */
   int xinteger[widthOut];
   float dx = (float)(widthIn - 1) / (float)(widthOut - 1);

   for (int kx = 0; kx < widthOut; kx++) {
      float x      = dx * (float)kx;
      xinteger[kx] = (int)nearbyintf(x);
   }

   int yinteger[heightOut];
   float dy = (float)(heightIn - 1) / (float)(heightOut - 1);

   for (int ky = 0; ky < heightOut; ky++) {
      float y      = dy * (float)ky;
      yinteger[ky] = (int)nearbyintf(y);
   }

   for (int ky = 0; ky < heightOut; ky++) {
      float yfetch = yinteger[ky];
      for (int kx = 0; kx < widthOut; kx++) {
         int xfetch = xinteger[kx];
         for (int f = 0; f < numBands; f++) {
            int fetchIdx         = yfetch * yStrideIn + xfetch * xStrideIn + f * bandStrideIn;
            int outputIdx        = kIndex(kx, ky, f, widthOut, heightOut, numBands);
            bufferOut[outputIdx] = bufferIn[fetchIdx];
         }
      }
   }
}

void bicubicInterp(
      float const *bufferIn,
      int widthIn,
      int heightIn,
      int numBands,
      int xStrideIn,
      int yStrideIn,
      int bandStrideIn,
      float *bufferOut,
      int widthOut,
      int heightOut) {

   // Interpolation using bicubic convolution with a = -1
   // (following Octave image toolbox's imremap function - change this?)
   float xinterp[widthOut];
   int xinteger[widthOut];
   float xfrac[widthOut];
   float dx = (float)(widthIn - 1) / (float)(widthOut - 1);

   for (int kx = 0; kx < widthOut; kx++) {
      float x      = dx * (float)kx;
      xinterp[kx]  = x;
      float xfloor = floorf(x);
      xinteger[kx] = (int)xfloor;
      xfrac[kx]    = x - xfloor;
   }

   float yinterp[heightOut];
   int yinteger[heightOut];
   float yfrac[heightOut];
   float dy = (float)(heightIn - 1) / (float)(heightOut - 1);

   for (int ky = 0; ky < heightOut; ky++) {
      float y      = dy * (float)ky;
      yinterp[ky]  = y;
      float yfloor = floorf(y);
      yinteger[ky] = (int)yfloor;
      yfrac[ky]    = y - yfloor;
   }

   memset(bufferOut, 0, sizeof(*bufferOut) * size_t(widthOut * heightOut * numBands));

   for (int xOff = 2; xOff > -2; xOff--) {
      for (int yOff = 2; yOff > -2; yOff--) {
         for (int ky = 0; ky < heightOut; ky++) {
            float ycoeff = bicubic(yfrac[ky] - (float)yOff);
            int yfetch   = yinteger[ky] + yOff;

            if (yfetch < 0)
               yfetch = -yfetch;
            if (yfetch >= heightIn)
               yfetch = heightIn - (yfetch - heightIn) - 1;

            for (int kx = 0; kx < widthOut; kx++) {
               float xcoeff = bicubic(xfrac[kx] - (float)xOff);
               int xfetch   = xinteger[kx] + xOff;

               if (xfetch < 0)
                  xfetch = -xfetch;
               if (xfetch >= widthIn)
                  xfetch = widthIn - (xfetch - widthIn) - 1;

               assert(xfetch >= 0 && xfetch < widthIn && yfetch >= 0 && yfetch < heightIn);

               for (int f = 0; f < numBands; f++) {
                  int fetchIdx  = yfetch * yStrideIn + xfetch * xStrideIn + f * bandStrideIn;
                  float p       = bufferIn[fetchIdx];
                  int outputIdx = kIndex(kx, ky, f, widthOut, heightOut, numBands);
                  bufferOut[outputIdx] += xcoeff * ycoeff * p;
               }
            }
         }
      }
   }
}
} // End anonymous namespace

// Rescale a buffer, preserving aspect ratio
void rescale(
      Buffer<float> &buffer,
      int newWidth,
      int newHeight,
      enum RescaleMethod rescaleMethod,
      enum InterpolationMethod interpMethod,
      enum Buffer<float>::Anchor anchor) {
   float xRatio       = (float)newWidth / buffer.getWidth();
   float yRatio       = (float)newHeight / buffer.getHeight();
   int resizedWidth   = newWidth;
   int resizedHeight  = newHeight;
   float resizeFactor = 1.0f;

   switch (rescaleMethod) {
      case CROP: resizeFactor = xRatio < yRatio ? yRatio : xRatio; break;
      case PAD: resizeFactor  = xRatio < yRatio ? xRatio : yRatio; break;
   }

   resizedWidth  = (int)nearbyintf(resizeFactor * buffer.getWidth());
   resizedHeight = (int)nearbyintf(resizeFactor * buffer.getHeight());

   std::vector<float> rawInput = buffer.asVector();
   std::vector<float> scaledInput(resizedWidth * resizedHeight * buffer.getFeatures());
   switch (interpMethod) {
      case BICUBIC:
         bicubicInterp(
               rawInput.data(),
               buffer.getWidth(),
               buffer.getHeight(),
               buffer.getFeatures(),
               buffer.getFeatures(),
               buffer.getFeatures() * buffer.getWidth(),
               1,
               scaledInput.data(),
               resizedWidth,
               resizedHeight);
         break;
      case NEAREST:
         nearestNeighborInterp(
               rawInput.data(),
               buffer.getWidth(),
               buffer.getHeight(),
               buffer.getFeatures(),
               buffer.getFeatures(),
               buffer.getFeatures() * buffer.getWidth(),
               1,
               scaledInput.data(),
               resizedWidth,
               resizedHeight);
         break;
   }
   buffer.set(scaledInput, resizedWidth, resizedHeight, buffer.getFeatures());

   // This final call resizes the buffer to our specified
   // newWidth and newHeight. If our rescaleMethod was PAD,
   // this actually grows the buffer to include the padded region.
   switch (rescaleMethod) {
      case CROP: buffer.crop(newWidth, newHeight, anchor); break;
      case PAD: buffer.grow(newWidth, newHeight, anchor); break;
   }
}
} // End BufferUtils namespace
} // End PV namespace
