#include "Buffer.hpp"
#include "utils/conversions.h"

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

      for(int v = 0; v < vector.size(); ++v) {
         int r = v / (cols * features);
         int c = v % (cols * features);
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

   void Buffer::resize(int rows, int columns, int features) {
      mData.resize(rows);
      for(auto& row : mData) {
         row.resize(columns);
         for(auto& column : row) {
            column.resize(features, 0.0f);
         }
      }
   }

   void Buffer::rescale(int targetRows, int targetColumns, enum RescaleMethod rescaleMethod, enum InterpolationMethod interpMethod) {
      float xRatio = static_cast<float>(targetColumns) / getColumns();
      float yRatio = static_cast<float>(targetRows) / getRows();

      int resizedRows, resizedColumns;
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
         resizedColumns = (int) nearbyintf(resizeFactor * getColumns()); 
         resizedRows = (int) nearbyintf(resizeFactor * getRows());
      }
      else {
         throw;
      }
      std::vector<float> rawInput = asVector();
      std::vector<float> scaledInput(rawInput.size());
      switch(interpMethod) {
      case BICUBIC:
         bicubicInterp(&rawInput[0], getColumns(), getRows(), getFeatures(), getFeatures(), getFeatures() * getColumns(), 1, &scaledInput[0], resizedColumns, resizedRows);
         break;
      case NEAREST:
         nearestNeighborInterp(&rawInput[0], getColumns(), getRows(), getFeatures(), getFeatures(), getFeatures() * getColumns(), 1, &scaledInput[0], resizedColumns, resizedRows);
         break;
      default:
         throw;
         break;
      }
      resize(resizedRows, resizedColumns, getFeatures());
      set(scaledInput);
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
