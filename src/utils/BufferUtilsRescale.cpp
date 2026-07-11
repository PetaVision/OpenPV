#include "BufferUtilsRescale.hpp"
#include "utils/conversions.hpp"
#include "utils/PVLog.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

namespace PV {
namespace BufferUtils {
namespace { // Anonymous namespace for "private" functions

inline static float bicubic(float x) {
   float const absx = std::fabs(x);
   return absx < 1 ? 1 + absx * absx * (-2 + absx) : absx < 2 ? 4 + absx * (-8 + absx * (5 - absx))
                                                              : 0;
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

   bool badDimensions = false;
   if (widthIn <= 0 or heightIn <= 0 or widthOut <= 0 or heightOut <= 0) {
      badDimensions = true;
      ErrorLog().printf(
            "nearest neighbor interpolation called with bad grid size:\n"
            "        input width = %d, input height = %d, output width = %d, output height = %d\n"
            "        Each of these values must be positive.\n",
            widthIn, heightIn, widthOut, heightOut);
   }
   if (widthOut == 1 and widthIn > 1) {
      badDimensions = true;
      ErrorLog().printf(
            "nearest neighbor interpolation called with input width 1 and output width %d\n"
            "        If input width is 1, output width must also be one.\n",
            widthOut);
   }
   if (heightOut == 1 and heightIn > 1) {
      badDimensions = true;
      ErrorLog().printf(
            "nearest neighbor interpolation called with input height 1 and output height %d\n"
            "        If input height is 1, output height must also be one.\n",
            widthOut);
   }
   FatalIf(badDimensions, "Bad arguments passed to nearest neighbor interpolation\n");

   /* Interpolation using nearest neighbor interpolation */
   int xinteger[widthOut];
   if (widthOut == 1) {
      assert(widthIn == 1); // Should hit a fatal error above if this assert fails.
      xinteger[0] = 0;
   }
   else {
      assert(widthOut > 1 and widthIn > 0); // Checked by FatalIf() above.
      float dx = static_cast<float>(widthIn - 1) / static_cast<float>(widthOut - 1);
      for (int kx = 0; kx < widthOut; kx++) {
         float x      = dx * static_cast<float>(kx);
         xinteger[kx] = static_cast<int>(std::nearbyint(x));
      }
   }

   int yinteger[heightOut];
   if (heightOut == 1) {
      assert(heightIn == 1); // Checked by FatalIf() above.
      yinteger[0] = 0;
   }
   else {
      assert(heightOut > 1 and heightIn > 0); // Checked by FatalIf() above.
      float dy = static_cast<float>(heightIn - 1) / static_cast<float>(heightOut - 1);
      for (int ky = 0; ky < heightOut; ky++) {
         float y      = dy * static_cast<float>(ky);
         yinteger[ky] = static_cast<int>(std::nearbyint(y));
      }
   }

   for (int ky = 0; ky < heightOut; ky++) {
      float yfetch = yinteger[ky];
      for (int kx = 0; kx < widthOut; kx++) {
         int xfetch = xinteger[kx];
         for (int f = 0; f < numBands; f++) {
            long fetchIdx =
                  yfetch * (long)yStrideIn + xfetch * (long)xStrideIn + f * (long)bandStrideIn;
            long outputIdx       = kIndex(kx, ky, f, widthOut, heightOut, numBands);
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

   FatalIf(
         widthIn <= 1 or heightIn <= 1 or widthOut <= 1 or heightOut <= 1,
         "bicubic interpolation called with too small a number of points:\n"
         "        input width = %d, input height = %d, output width = %d, output height = %d\n"
         "        Each of these values must be 2 or greater.\n",
         widthIn, heightIn, widthOut, heightOut);

   // Interpolation using bicubic convolution with a = -1
   // (following Octave image toolbox's imremap function - change this?)
   int xinteger[widthOut];
   float xfrac[widthOut];
   float dx = (float)(widthIn - 1) / (float)(widthOut - 1);

   for (int kx = 0; kx < widthOut; kx++) {
      float x      = dx * (float)kx;
      float xfloor = std::floor(x);
      xinteger[kx] = static_cast<int>(xfloor);
      xfrac[kx]    = x - xfloor;
   }

   int yinteger[heightOut];
   float yfrac[heightOut];
   float dy = (float)(heightIn - 1) / (float)(heightOut - 1);

   for (int ky = 0; ky < heightOut; ky++) {
      float y      = dy * (float)ky;
      float yfloor = std::floor(y);
      yinteger[ky] = static_cast<int>(yfloor);
      yfrac[ky]    = y - yfloor;
   }

   // We increment-add as we interpolate, so we need to set the output buffer to zero at the start.
   std::fill_n(bufferOut, widthOut * heightOut * numBands, 0.0f);

   for (int xOff = 2; xOff > -2; xOff--) {
      for (int yOff = 2; yOff > -2; yOff--) {
         for (int ky = 0; ky < heightOut; ky++) {
            float ycoeff = bicubic(yfrac[ky] - (float)yOff);
            int yfetch   = yinteger[ky] + yOff;

            if (yfetch < 0) { yfetch = -yfetch; }
            if (yfetch >= heightIn) { yfetch = heightIn - (yfetch - heightIn) - 1; }

            for (int kx = 0; kx < widthOut; kx++) {
               float xcoeff = bicubic(xfrac[kx] - (float)xOff);
               int xfetch   = xinteger[kx] + xOff;

               if (xfetch < 0) { xfetch = -xfetch; }
               if (xfetch >= widthIn) { xfetch = widthIn - (xfetch - widthIn) - 1; }

               assert(xfetch >= 0 && xfetch < widthIn && yfetch >= 0 && yfetch < heightIn);

               for (int f = 0; f < numBands; f++) {
                  long fetchIdx =
                        yfetch * (long)yStrideIn + xfetch * (long)xStrideIn + f * bandStrideIn;
                  float p        = bufferIn[fetchIdx];
                  long outputIdx = kIndex(kx, ky, f, widthOut, heightOut, numBands);
                  bufferOut[outputIdx] += xcoeff * ycoeff * p;
               }
            }
         }
      }
   }
}
} // End anonymous namespace

// Rescale a buffer, to new width and height. First the buffer is stretched or shrunk while
// preserving the aspect ratio as closely as possible, so that one of the dimensions agrees with
// the desired value. Then the buffer is cropped or padded to make the other dimension agree.
void rescale(
      Buffer<float> &buffer,
      int newWidth,
      int newHeight,
      enum RescaleMethod rescaleMethod,
      enum InterpolationMethod interpMethod,
      enum Buffer<float>::Anchor anchor) {

   // If newWidth and newHeight are the existing dimensions, return as nothing needs to be done.
   if (newWidth == buffer.getWidth() and newHeight == buffer.getHeight()) {
      return;
   }

   float xRatio       = (float)newWidth / buffer.getWidth();
   float yRatio       = (float)newHeight / buffer.getHeight();
   int resizedWidth   = newWidth;
   int resizedHeight  = newHeight;
   float resizeFactor = 1.0f;

   switch (rescaleMethod) {
      case CROP: resizeFactor = std::max(xRatio, yRatio); break;
      case PAD: resizeFactor  = std::min(xRatio, yRatio); break;
   }

   resizedWidth  = static_cast<int>(std::nearbyint(resizeFactor * buffer.getWidth()));
   resizedHeight = static_cast<int>(std::nearbyint(resizeFactor * buffer.getHeight()));
   if (resizedWidth != buffer.getWidth() or resizedHeight != buffer.getHeight()) {
      std::vector<float> rawInput = buffer.asVector();
      long resizedNumElements =
            (long)resizedWidth * (long)resizedHeight * (long)buffer.getFeatures();
      std::vector<float> scaledInput(resizedNumElements);
      switch (interpMethod) {
         case BICUBIC:
            bicubicInterp(
                  rawInput.data(),
                  buffer.getWidth(),
                  buffer.getHeight(),
                  buffer.getFeatures(),
                  buffer.getFeatures(),
                  buffer.getFeatures() * buffer.getWidth(),
                  1 /*stride in feature dimension*/,
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
                  1 /*stride in feature dimension*/,
                  scaledInput.data(),
                  resizedWidth,
                  resizedHeight);
            break;
         default:
            Fatal().printf("Unrecognized interpolation method %d\n", interpMethod);
            break;
      }
      buffer.set(scaledInput, resizedWidth, resizedHeight, buffer.getFeatures());
   }

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
