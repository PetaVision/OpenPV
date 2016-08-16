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

   // Resizing a Buffer will clear its contents. Use rescale or crop to preserve values.
   // Is there a better name for this? clearAndResize()?
   void Buffer::resize(int width, int height, int features) {
      mData.clear();
      mData.resize(height * width * features);
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

   int Buffer::getOffsetY(enum OffsetAnchor offsetAnchor, int offsetY, int newHeight, int currentHeight) {
      switch(offsetAnchor) {
         case NORTHWEST:
         case NORTH:
         case NORTHEAST:
            return offsetY;
         case WEST:
         case CENTER:
         case EAST:
            return currentHeight/2 - newHeight/2 + offsetY;
         case SOUTHWEST:
         case SOUTH:
         case SOUTHEAST:
         return offsetY + currentHeight - newHeight;
      }
      return 0;
   }

   // TODO: Some of these methods could be moved to a BufferManipulaton class, or maybe
   // a TransformableBuffer subclass? RescaleableBuffer? MalleableBuffer?
   bool Buffer::constrainPoint(int &x, int &y, int minX, int maxX, int minY, int maxY, enum PointConstraintMethod method) {
      bool moved_x = x < minX || y > maxX;
      bool moved_y = y < minY || y > maxY;
      if (moved_x) {
         assert(minX <= maxX);
         int sizeX = maxX-minX;
         int newX = x; 
         switch (method) {
            case MIRROR:
               newX -= minX;
               newX %= (2 * (sizeX + 1));
               if (newX < 0) ++newX;
               newX = abs(newX);
               if (newX > sizeX) newX = 2*sizeX + 1 - newX;
               newX += minX;
               break;
            case CLAMP:
               if (newX < minX) newX = minX;
               if (newX > maxX) newX = maxX;
               break;
            case WRAP:
               newX -= minX;
               newX %= sizeX + 1;
               if (newX < 0) newX += sizeX + 1;
               newX += minX;
               break;
         }
         assert(newX >= minX && newX <= maxX);
         x = newX;
      }
      if (moved_y) {
         assert(minY <= maxY);
         int sizeY = maxY - minY;
         int newY = y;
         switch (method) {
         case MIRROR:
            newY -= minY;
            newY %= (2 * (sizeY + 1));
            if (newY < 0) ++newY;
            newY = abs(newY);
            if (newY > sizeY) newY = 2*sizeY + 1 - newY;
            newY += minY;
            break;
         case CLAMP:
            if (newY < minY) newY = minY;
            if (newY > maxY) newY = maxY;
            break;
         case WRAP:
            newY -= minY;
            newY %= sizeY + 1;
            if (newY < 0) newY += sizeY + 1;
            newY += minY;
            break;
         }
         assert(newY >= minY && newY <= maxY);
         y = newY;
      }
      return moved_x || moved_y;
   }

   // Crops a buffer down to the specified size, OR grows the canvas keeping the original in the middle (how does this interact with offset?)
   void Buffer::crop(int targetWidth, int targetHeight, enum OffsetAnchor offsetAnchor, int offsetX, int offsetY) {
      int smallerWidth  = targetWidth  < getWidth()  ? targetWidth  : getWidth();
      int smallerHeight = targetHeight < getHeight() ? targetHeight : getHeight();
      int biggerWidth   = targetWidth  > getWidth()  ? targetWidth  : getWidth();
      int biggerHeight  = targetHeight > getHeight() ? targetHeight : getHeight();
      int originX = 0;
      int originY = 0;
       // If we're "cropping" to a larger canvas, place the original in the center
      if(targetHeight > getHeight()) {
         originY = (targetHeight - getHeight()) / 2;
      }
      if(targetWidth > getWidth()) {
         originX = (targetWidth - getWidth()) / 2;
      }
 

     offsetX = getOffsetX(offsetAnchor, offsetX-originX, smallerWidth,  biggerWidth);
     offsetY = getOffsetY(offsetAnchor, offsetY-originY, smallerHeight, biggerHeight);
     Buffer cropped(targetWidth, targetHeight, getFeatures());

     for(int smallY = 0; smallY < smallerHeight; ++smallY) {
         for(int smallX = 0; smallX < smallerWidth; ++smallX) {
            for(int f = 0; f < getFeatures(); ++f) {
               int x = smallX + offsetX, y = smallY + offsetY;
               constrainPoint(x, y, 0, getWidth()-1, 0, getHeight()-1, CLAMP);
               cropped.set(smallX+originX, smallY+originY, f, at(x, y, f));
            }
         }
      }
      set(cropped.asVector(), targetWidth, targetHeight, getFeatures());
   }

   void Buffer::rescale(int targetWidth, int targetHeight, enum RescaleMethod rescaleMethod, enum InterpolationMethod interpMethod) {
      float xRatio = static_cast<float>(targetWidth) / getWidth();
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

      // This final call to crop resizes the buffer to our specified
      // targetWidth and targetHeight. If our rescaleMethod was PAD,
      // this actually grows the buffer to include the padded region.
      // TODO: Accept offsetAnchor as an argument to use here?
      crop(targetWidth, targetHeight, CENTER, 0, 0);
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
