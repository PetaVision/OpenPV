#include "RealBuffer.hpp"
#include "conversions.h"
#include <cstring>
#include <cmath>


namespace PV {

// Rescale a buffer, preserving aspect ratio
void RealBuffer::rescale(int newWidth, int newHeight,
      enum RescaleMethod rescaleMethod,
      enum InterpolationMethod interpMethod,
      enum Anchor anchor) {
   float xRatio = static_cast<float>(newWidth)  / getWidth();
   float yRatio = static_cast<float>(newHeight) / getHeight();
   int resizedWidth = newWidth;
   int resizedHeight = newHeight;
   float resizeFactor = 1.0f;

   switch(rescaleMethod) {
      case CROP:
         resizeFactor = xRatio < yRatio ? yRatio : xRatio;
         break;
      case PAD:
         resizeFactor = xRatio < yRatio ? xRatio : yRatio;
         break;
   }

   resizedWidth   = static_cast<int>(nearbyintf(resizeFactor * getWidth()));
   resizedHeight  = static_cast<int>(nearbyintf(resizeFactor * getHeight()));

   std::vector<float> rawInput = asVector();
   std::vector<float> scaledInput(resizedWidth * resizedHeight * getFeatures());
   switch(interpMethod) {
      case BICUBIC:
         bicubicInterp(rawInput.data(),                        // Input data
               getWidth(),    getHeight(), getFeatures(),      // Input dimensions
               getFeatures(), getFeatures() * getWidth(), 1,   // Strides
               scaledInput.data(),                             // Output data
               resizedWidth,  resizedHeight);                  // Target dimensions
         break;
      case NEAREST:
         nearestNeighborInterp(rawInput.data(),                // Input data
               getWidth(),    getHeight(), getFeatures(),      // Input dimensions
               getFeatures(), getFeatures() * getWidth(), 1,   // Strides
               scaledInput.data(),                             // Output data
               resizedWidth,  resizedHeight);                  // Target dimensions
         break;
   }
   set(scaledInput, resizedWidth, resizedHeight, getFeatures());

   // This final call resizes the buffer to our specified
   // newWidth and newHeight. If our rescaleMethod was PAD,
   // this actually grows the buffer to include the padded region.
   switch(rescaleMethod) {
      case CROP:
         crop(newWidth, newHeight, anchor);
         break;
      case PAD:
         grow(newWidth, newHeight, anchor);
         break;
   }
}

void RealBuffer::nearestNeighborInterp(float const * bufferIn,
      int widthIn,   int heightIn,  int numBands,
      int xStrideIn, int yStrideIn, int bandStrideIn,
      float * bufferOut,
      int widthOut,  int heightOut) {

   /* Interpolation using nearest neighbor interpolation */
   int xinteger[widthOut];
   float dx = (float) (widthIn-1)/(float) (widthOut-1);

   for (int kx=0; kx<widthOut; kx++) {
      float x = dx * (float) kx;
      xinteger[kx] = (int) nearbyintf(x);
   }

   int yinteger[heightOut];
   float dy = (float) (heightIn-1)/(float) (heightOut-1);
   
   for (int ky=0; ky<heightOut; ky++) {
      float y = dy * (float) ky;
      yinteger[ky] = (int) nearbyintf(y);
   }

   for (int ky=0; ky<heightOut; ky++) {
      float yfetch = yinteger[ky];
      for (int kx=0; kx<widthOut; kx++) {
         int xfetch = xinteger[kx];
         for (int f=0; f<numBands; f++) {
            int fetchIdx = yfetch * yStrideIn + xfetch * xStrideIn + f * bandStrideIn;
            int outputIdx = kIndex(kx, ky, f, widthOut, heightOut, numBands);
            bufferOut[outputIdx] = bufferIn[fetchIdx];
         }
      }
   }
}

void RealBuffer::bicubicInterp(float const * bufferIn,
      int widthIn,   int heightIn,  int numBands,
      int xStrideIn, int yStrideIn, int bandStrideIn,
      float * bufferOut,
      int widthOut,  int heightOut) {

   /* Interpolation using bicubic convolution with a=-1 (following Octave image toolbox's imremap function -- change this?). */
   float xinterp[widthOut];
   int xinteger[widthOut];
   float xfrac[widthOut];
   float dx = (float) (widthIn-1)/(float) (widthOut-1);

   for (int kx=0; kx<widthOut; kx++) {
      float x = dx * (float) kx;
      xinterp[kx] = x;
      float xfloor = floorf(x);
      xinteger[kx] = (int) xfloor;
      xfrac[kx] = x-xfloor;
   }

   float yinterp[heightOut];
   int yinteger[heightOut];
   float yfrac[heightOut];
   float dy = (float) (heightIn-1)/(float) (heightOut-1);

   for (int ky=0; ky<heightOut; ky++) {
      float y = dy * (float) ky;
      yinterp[ky] = y;
      float yfloor = floorf(y);
      yinteger[ky] = (int) yfloor;
      yfrac[ky] = y-yfloor;
   }

   memset(bufferOut, 0, sizeof(*bufferOut)*size_t(widthOut*heightOut*numBands));

   for (int xOff = 2; xOff > -2; xOff--) {
      for (int yOff = 2; yOff > -2; yOff--) {
         for (int ky=0; ky<heightOut; ky++) {
            float ycoeff = bicubic(yfrac[ky]-(float) yOff);
            int yfetch = yinteger[ky]+yOff;
            
            if (yfetch < 0) yfetch = -yfetch;
            if (yfetch >= heightIn) yfetch = heightIn - (yfetch - heightIn) - 1;

            for (int kx=0; kx<widthOut; kx++) {
               float xcoeff = bicubic(xfrac[kx]-(float) xOff);
               int xfetch = xinteger[kx]+xOff;

               if (xfetch < 0) xfetch = -xfetch;
               if (xfetch >= widthIn) xfetch = widthIn - (xfetch - widthIn) - 1;

               assert(xfetch >= 0 && xfetch < widthIn && yfetch >= 0 && yfetch < heightIn);

               for (int f=0; f<numBands; f++) {
                  int fetchIdx = yfetch * yStrideIn + xfetch * xStrideIn + f * bandStrideIn;
                  float p = bufferIn[fetchIdx];
                  int outputIdx = kIndex(kx, ky, f, widthOut, heightOut, numBands);
                  bufferOut[outputIdx] += xcoeff * ycoeff * p;
               }
            }
         }
      }
   }
}


}
