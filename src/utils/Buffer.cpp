#include "Buffer.hpp"
#include "conversions.h"
#include "PVLog.hpp"

#include <cstring>
#include <cmath>

namespace PV {

   Buffer::Buffer(int rows, int columns, int features) {
      resize(rows, columns, features);
   }

   Buffer::Buffer() {
      resize(1,1,1);
   }

   float Buffer::at(int row, int column, int feature) {
      return mData.at(row).at(column).at(feature);
   }

   void Buffer::set(int row, int column, int feature, float value) {
      mData.at(row).at(column).at(feature) = value;
   }

   void Buffer::set(const std::vector<float> &vector) {
      int rows = getRows();
      int cols = getColumns();
      int features = getFeatures();

      pvErrorIf(vector.size() != rows * cols * features,
            "Invalid vector size: Expected %d elements, vector contained %d elements. Did you remember to call resize() before set()?\n",
            rows * cols * features, vector.size());

      for(int v = 0; v < vector.size(); ++v) {
         int r = v / (cols * features);
         int c = (v / features) % cols;
         int f = v % features;
         set(r, c, f, vector.at(v));
      }
   }

   std::vector<float> Buffer::asVector() {
      std::vector<float> result(getRows()*getColumns()*getFeatures());
      int v = 0;
      for(auto row : mData) {
         for(auto col : row) {
            for(auto feature : col) {
               result.at(v++) = feature;
            }
         }
      }
      return result;
   }

   // Resizing a Buffer will clear its contents. Use rescale or crop to preserve values.
   void Buffer::resize(int rows, int columns, int features) {
      mData.resize(rows);
      for(int r = 0; r < rows; ++r) {
         mData.at(r).resize(columns);
         for(int c = 0; c < columns; ++c) {
            mData.at(r).at(c).clear();
            mData.at(r).at(c).resize(features, 0.0f);
         }
      }
   }

   //Offsets based on an anchor point, so calculate offsets based off a given anchor point
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

   //TODO: Some of these methods could be moved to a BufferManipulaton class
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

   void Buffer::crop(int targetRows, int targetColumns, enum OffsetAnchor offsetAnchor, int offsetX, int offsetY) {
      int smallerRows = targetRows < getRows() ? targetRows : getRows();
      int smallerCols = targetColumns < getColumns() ? targetColumns : getColumns();
      int biggerRows = targetRows > getRows() ? targetRows : getRows();
      int biggerCols = targetColumns > getColumns() ? targetColumns : getColumns();
      int originX = 0;
      int originY = 0;
       // If we're "cropping" to a larger canvas, place the original in the center
      if(targetRows > getRows()) {
         originY = (targetRows - getRows()) / 2;
      }
      if(targetColumns > getColumns()) {
         originX = (targetColumns - getColumns()) / 2;
      }
 

     offsetX = getOffsetX(offsetAnchor, offsetX-originX, smallerCols, biggerCols);
     offsetY = getOffsetY(offsetAnchor, offsetY-originY, smallerRows, biggerRows);
     Buffer cropped(targetRows, targetColumns, getFeatures());

     for(int r = 0; r < smallerRows; ++r) {
         for(int c = 0; c < smallerCols; ++c) {
            for(int f = 0; f < getFeatures(); ++f) {
               int x = c + offsetX, y = r + offsetY;
               constrainPoint(x, y, 0, getColumns()-1, 0, getRows()-1, CLAMP);
               cropped.set(r+originY, c+originX, f, at(y, x, f));
            }
         }
      }
      resize(targetRows, targetColumns, getFeatures());
      set(cropped.asVector());
   }

   void Buffer::rescale(int targetRows, int targetColumns, enum RescaleMethod rescaleMethod, enum InterpolationMethod interpMethod) {
      float xRatio = static_cast<float>(targetColumns) / getColumns();
      float yRatio = static_cast<float>(targetRows) / getRows();

      int resizedRows = targetRows;
      int resizedColumns = targetColumns;
      float resizeFactor = 1.0f;

      if (rescaleMethod == CROP) {
         resizeFactor = xRatio < yRatio ? yRatio : xRatio;
         // resizeFactor * width should be >= getLayerLoc()->nx; resizeFactor * height should be >= getLayerLoc()->ny,
         // and one of these relations should be == (up to floating-point roundoff).
         resizedColumns = static_cast<int>(nearbyintf(resizeFactor * getColumns()));
         resizedRows = static_cast<int>(nearbyintf(resizeFactor * getRows()));
      }
      else if (rescaleMethod == PAD) {
         resizeFactor = xRatio < yRatio ? xRatio : yRatio;
         // resizeFactor * width should be <= getLayerLoc()->nx; resizeFactor * height should be <= getLayerLoc()->ny,
         // and one of these relations should be == (up to floating-point roundoff).
         resizedColumns = static_cast<int>(nearbyintf(resizeFactor * getColumns()));
         resizedRows = static_cast<int>(nearbyintf(resizeFactor * getRows()));
      }

      std::vector<float> rawInput = asVector();
      std::vector<float> scaledInput(resizedRows * resizedColumns * getFeatures());
      switch(interpMethod) {
         case BICUBIC:
            bicubicInterp(rawInput.data(), getColumns(), getRows(), getFeatures(), getFeatures(), getFeatures() * getColumns(), 1, scaledInput.data(), resizedColumns, resizedRows);
            break;
         case NEAREST:
            nearestNeighborInterp(rawInput.data(), getColumns(), getRows(), getFeatures(), getFeatures(), getFeatures() * getColumns(), 1, scaledInput.data(), resizedColumns, resizedRows);
            break;
      }
      resize(resizedRows, resizedColumns, getFeatures());
      set(scaledInput);

      // This final call to crop resizes the buffer to our specified
      // targetRows and targetColumns. If our rescaleMethod was PAD,
      // this actually grows the buffer to include the padded region.
      // TODO: Accept offsetAnchor as an argument to use here
      crop(targetRows, targetColumns, CENTER, 0, 0);
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
