#include "Image.hpp"
#include "PVLog.hpp"

# ifndef STB_IMAGE_IMPLEMENTATION
#  define STB_IMAGE_IMPLEMENTATION 
#  include "stb_image.h"
# endif

namespace PV {

   // TODO: Image should probably have Image::load(filename) and this constructor should call it
   Image::Image(std::string filename) {
      int width = 0, height = 0, channels = 0;
      uint8_t* data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
      pvErrorIf(data == nullptr, " File not found: %s\n", filename.c_str());
      resize(height, width, channels);

      for(int row = 0; row < height; ++row) {
         for(int col = 0; col < width; ++col) {
            for(int f = 0; f < channels; ++f) {
               float value = static_cast<float>(data[(row * width + col) * channels + f]) / 255.0f;
               set(row, col, f, value);
            }
         }
      }
      
      stbi_image_free(data);
   }

   Image::Image(const std::vector<float> &data, int width, int height, int channels) {
      resize(height, width, channels);
      set(data);
   }

   void Image::setPixel(int x, int y, float r, float g, float b) {
      if(getFeatures() > mRPos) {
         set(y, x, mRPos, r);
      }
      if(getFeatures() > mGPos) {
         set(y, x, mGPos, g);
      }
      if(getFeatures() > mBPos) {
         set(y, x, mBPos, b);
      }
   }

   void Image::setPixel(int x, int y, float r, float g, float b, float a) {
      setPixel(x, y, r, g, b);
      set(y, x, mAPos, a);
   }

   float Image::getPixelR(int x, int y) {
      if(getFeatures() <= mRPos) {
         return 0.0f;
      }
      return at(y, x, mRPos); 
   }

   float Image::getPixelG(int x, int y) {
      if(getFeatures() <= mGPos) {
         return 0.0f;
      }
      return at(y, x, mGPos); 
   }
   
   float Image::getPixelB(int x, int y) {
      if(getFeatures() <= mBPos) {
         return 0.0f;
      }
      return at(y, x, mBPos); 
   }

   float Image::getPixelA(int x, int y) {
      if(getFeatures() <= mAPos) {
         return 1.0f;
      }
      return at(y, x, mAPos);
   }

   void Image::convertToGray(bool alphaChannel) {
      if(getFeatures() < 3) {
         if((getFeatures() == 1 && !alphaChannel) || (getFeatures() == 2 && alphaChannel)) {
            // Do nothing if we are already in the correct format
            return;
         }
         else {
           // We are already grayscale, but we're adding or removing an alpha channel
           Buffer grayScale(getRows(), getColumns(), alphaChannel ? 2 : 1);

            for(int r = 0; r < getRows(); ++r) {
               for(int c = 0; c < getColumns(); ++c) {
                  grayScale.set(r, c, 0, at(r, c, 0));
                  if(alphaChannel) {
                     grayScale.set(r, c, 1, 1.0f);
                  }
               }
            }

            resize(getRows(), getColumns(), alphaChannel ? 2 : 1);
            set(grayScale.asVector());
            return;
         }
      }
      else {
         // We're currently RGB or RGBA and need to be Grayscale or Grayscale + Alpha
         // RGB weights from <https://en.wikipedia.org/wiki/Grayscale>, citing Pratt, Digital Image Processing
         const float rgbWeights[3] = { mRToGray, mGToGray, mBToGray };//{0.30f, 0.59f, 0.11f};
         Buffer grayScale(getRows(), getColumns(), alphaChannel ? 2 : 1);

         for(int r = 0; r < getRows(); ++r) {
            for(int c = 0; c < getColumns(); ++c) {
               float sum = 0.0f;
               for(int f = 0; f < 3; ++f) {
                  sum += at(r, c, f) * rgbWeights[f];
               }
               grayScale.set(r, c, 0, sum);
               if(alphaChannel) {
                  if(getFeatures() > 3) {
                     grayScale.set(r, c, 1, at(r, c, 3));
                  }
                  else {
                     grayScale.set(r, c, 1, 1.0f);
                  }
               }
            }
         }

         resize(getRows(), getColumns(), alphaChannel ? 2 : 1);
         set(grayScale.asVector());
      }
   }

   void Image::convertToColor(bool alphaChannel) {
      // Are we already color? If so, do we need to add or remove an alpha channel?
      if(getFeatures() > 2) {
         if((getFeatures() == 3 && !alphaChannel) || (getFeatures() == 4 && alphaChannel)) {
            // This is the correct format already, nothing to be done
            return;
         }
         else {
            // We're already color, but we're adding or removing an alpha channel
            Buffer color(getRows(), getColumns(), alphaChannel ? 4 : 3);
            for(int r = 0; r < getRows(); ++r) {
               for(int c = 0; c < getColumns(); ++c) {
                  color.set(r, c, mRPos, at(r, c, mRPos));
                  color.set(r, c, mGPos, at(r, c, mGPos));
                  color.set(r, c, mBPos, at(r, c, mBPos));
                  if(alphaChannel) { 
                     color.set(r, c, mAPos, 1.0f);
                  }
               }
            }
            resize(getRows(), getColumns(), alphaChannel ? 4 : 3);
            set(color.asVector());
         }
      }
      else {
         // We're converting a grayscale image to color
         Buffer color(getRows(), getColumns(), alphaChannel ? 4 : 3);
         for(int r = 0; r < getRows(); ++r) {
            for(int c = 0; c < getColumns(); ++c) {
               float val = at(r, c, 0);
               color.set(r, c, mRPos, val);
               color.set(r, c, mGPos, val);
               color.set(r, c, mBPos, val);
               if(alphaChannel) { 
                  if(getFeatures() == 2) {
                     color.set(r, c, mAPos, at(r, c, 1));
                  }
                  else {
                     color.set(r, c, mAPos, 1.0f);
                  }
               }
            }
         }
         resize(getRows(), getColumns(), alphaChannel ? 4 : 3);
         set(color.asVector());
      }
   }
}
