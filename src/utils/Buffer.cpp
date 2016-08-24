#include "Buffer.hpp"
#include "conversions.h"
#include "PVLog.hpp"

#include <cstring>
#include <cmath>

namespace PV {

   Buffer::Buffer(int width, int height, int features) {
      resize(width, height, features);
   }

   Buffer::Buffer() {
      resize(1,1,1);
   }

   Buffer::Buffer(const std::vector<float> &data, int width, int height, int features) {
      set(data, width, height, features);
   }

   float Buffer::at(int x, int y, int feature) {
      return mData.at(index(x, y, feature));
   }

   void Buffer::set(int x, int y, int feature, float value) {
      mData.at(index(x, y, feature)) = value;
   }

   void Buffer::set(const std::vector<float> &vector, int width, int height, int features) {
      pvErrorIf(vector.size() != width * height * features,
         "Invalid vector size: Expected %d elements, vector contained %d elements.\n",
         width * height * features, vector.size());
      mData = vector;
      mWidth = width;
      mHeight = height;
      mFeatures = features;
   }

   const std::vector<float> Buffer::asVector() {
       return mData;
   }

   // Resizing a Buffer will clear its contents. Use rescale, crop, or grow to preserve values.
   void Buffer::resize(int width, int height, int features) {
      mData.clear();
      mData.resize(height * width * features, 0.0f);
      mWidth = width;
      mHeight = height;
      mFeatures = features;
   }

   int Buffer::getOffsetX(enum OffsetAnchor offsetAnchor, int offsetX, int newWidth, int currentWidth) {
      switch(offsetAnchor) {
         case NORTHWEST:
         case WEST:
         case SOUTHWEST:
            return offsetX;
         case NORTH:
         case CENTER:
         case SOUTH:
            return currentWidth/2 - newWidth/2 + offsetX;
         case NORTHEAST:
         case EAST:
         case SOUTHEAST:
            return offsetX + currentWidth - newWidth;
      }
      return 0;
   }

   int Buffer::getOffsetY(enum OffsetAnchor offsetAnchor, int offsetY, int smallerHeight, int biggerHeight) {
      switch(offsetAnchor) {
         case NORTHWEST:
         case NORTH:
         case NORTHEAST:
            return offsetY;
         case WEST:
         case CENTER:
         case EAST:
            return biggerHeight/2 - smallerHeight/2 + offsetY;
         case SOUTHWEST:
         case SOUTH:
         case SOUTHEAST:
         return offsetY + biggerHeight - smallerHeight;
      }
      return 0;
   }

   void Buffer::translate(int offsetX, int offsetY) {
      Buffer result(getWidth(), getHeight(), getFeatures());
      for(int y = 0; y < getHeight(); ++y) {
         for(int x = 0; x < getWidth(); ++x) {
            for(int f = 0; f < getFeatures(); ++f) {
               int destX = x + offsetX;
               int destY = y + offsetY;
               if(destX < 0 || destX >= getWidth())  continue;
               if(destY < 0 || destY >= getHeight()) continue;
               result.set(destX, destY, f, at(x, y, f));
            }
         }
      }
      set(result.asVector(), getWidth(), getHeight(), getFeatures());
   }
  
   // Grows a buffer
   void Buffer::grow(int newWidth, int newHeight, enum OffsetAnchor offsetAnchor) {
      if(newWidth < getWidth() && newHeight < getHeight()) {
         return;
      }
      int offsetX = getOffsetX(offsetAnchor, 0, getWidth(),  newWidth);
      int offsetY = getOffsetY(offsetAnchor, 0, getHeight(), newHeight);
      Buffer bigger(newWidth, newHeight, getFeatures());

      for(int y = 0; y < getHeight(); ++y) {
         for(int x = 0; x < getWidth(); ++x) {
            for(int f = 0; f < getFeatures(); ++f) {
               int destX = x + offsetX;
               int destY = y + offsetY;
               if(destX < 0 || destX >= newWidth)  continue;
               if(destY < 0 || destY >= newHeight) continue;
               bigger.set(destX, destY, f, at(x, y, f));
            }
         }
      }
      set(bigger.asVector(), newWidth, newHeight, getFeatures());
   }

   // Crops a buffer down to the specified size
   void Buffer::crop(int targetWidth, int targetHeight, enum OffsetAnchor offsetAnchor) {
      if(targetWidth >= getWidth() && targetHeight >= getHeight()) {
         return;
      }
      int offsetX = getOffsetX(offsetAnchor, 0, targetWidth,  getWidth());
      int offsetY = getOffsetY(offsetAnchor, 0, targetHeight, getHeight());
      Buffer cropped(targetWidth, targetHeight, getFeatures());

      for(int destY = 0; destY < targetHeight; ++destY) {
         for(int destX = 0; destX < targetWidth; ++destX) {
            for(int f = 0; f < getFeatures(); ++f) {
               int sourceX = destX + offsetX;
               int sourceY = destY + offsetY;
               if(sourceX < 0 || sourceX >= getWidth())  continue;
               if(sourceY < 0 || sourceX >= getHeight()) continue;
               cropped.set(destX, destY, f, at(sourceX, sourceY, f));
            }
         }
      }
      set(cropped.asVector(), targetWidth, targetHeight, getFeatures());
   }

   void Buffer::rescale(int targetWidth, int targetHeight, enum RescaleMethod rescaleMethod, enum InterpolationMethod interpMethod, enum OffsetAnchor offsetAnchor) {
      float xRatio = static_cast<float>(targetWidth)  / getWidth();
      float yRatio = static_cast<float>(targetHeight) / getHeight();
      int resizedWidth = targetWidth;
      int resizedHeight = targetHeight;
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
            bicubicInterp(rawInput.data(), getWidth(), getHeight(), getFeatures(), getFeatures(), getFeatures() * getWidth(), 1, scaledInput.data(), resizedWidth, resizedHeight);
            break;
         case NEAREST:
            nearestNeighborInterp(rawInput.data(), getWidth(), getHeight(), getFeatures(), getFeatures(), getFeatures() * getWidth(), 1, scaledInput.data(), resizedWidth, resizedHeight);
            break;
      }
      set(scaledInput, resizedWidth, resizedHeight, getFeatures());

      // This final call resizes the buffer to our specified
      // targetWidth and targetHeight. If our rescaleMethod was PAD,
      // this actually grows the buffer to include the padded region.
      switch(rescaleMethod) {
         case CROP:
            crop(targetWidth, targetHeight, offsetAnchor);
            break;
         case PAD:
            grow(targetWidth, targetHeight, offsetAnchor);
            break;
      }
   }


   void Buffer::nearestNeighborInterp(float const * bufferIn, int widthIn, int heightIn, int numBands, int xStrideIn, int yStrideIn, int bandStrideIn, float * bufferOut, int widthOut, int heightOut) {
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

   void Buffer::bicubicInterp(float const * bufferIn, int widthIn, int heightIn, int numBands, int xStrideIn, int yStrideIn, int bandStrideIn, float * bufferOut, int widthOut, int heightOut) {
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
