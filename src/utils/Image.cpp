#include "Image.hpp"
#include "PVLog.hpp"

// These defines are required by the stb headers
#ifndef STB_IMAGE_IMPLEMENTATION
#   define STB_IMAGE_IMPLEMENTATION 
#   include "stb_image.h"
#endif
#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#   define STB_IMAGE_WRITE_IMPLEMENTATION
#   include "stb_image_write.h"
#endif

namespace PV {

   Image::Image(std::string filename) {
      read(filename);
   }

   Image::Image(const std::vector<float> &data, int width, int height, int channels) {
      set(data, width, height, channels);
   }

   void Image::setPixel(int x, int y, float r, float g, float b) {
      if (getFeatures() > mRPos) {
         set(x, y, mRPos, r);
      }
      if (getFeatures() > mGPos) {
         set(x, y, mGPos, g);
      }
      if (getFeatures() > mBPos) {
         set(x, y, mBPos, b);
      }
   }

   void Image::setPixel(int x, int y, float r, float g, float b, float a) {
      setPixel(x, y, r, g, b);
      set(x, y, mAPos, a);
   }

   float Image::getPixelR(int x, int y) {
      if (getFeatures() <= mRPos) {
         return 0.0f;
      }
      return at(x, y, mRPos); 
   }

   float Image::getPixelG(int x, int y) {
      if (getFeatures() <= mGPos) {
         return 0.0f;
      }
      return at(x, y, mGPos); 
   }
   
   float Image::getPixelB(int x, int y) {
      if (getFeatures() <= mBPos) {
         return 0.0f;
      }
      return at(x, y, mBPos); 
   }

   float Image::getPixelA(int x, int y) {
      if (getFeatures() <= mAPos) {
         return 1.0f;
      }
      return at(x, y, mAPos);
   }

   void Image::convertToGray(bool alphaChannel) {
      if (getFeatures() < 3) {
         if ((getFeatures() == 1 && !alphaChannel) || (getFeatures() == 2 && alphaChannel)) {
            // Do nothing if we are already in the correct format
            return;
         }
         else {
            // We are already grayscale, but we're adding or removing an alpha channel
            Buffer grayScale(getWidth(), getHeight(), alphaChannel ? 2 : 1);
            for (int y = 0; y < getHeight(); ++y) {
               for (int x = 0; x < getWidth(); ++x) {
                  grayScale.set(x, y, 0, at(x, y, 0));
                  if (alphaChannel) {
                     grayScale.set(x, y, 1, 1.0f);
                  }
               }
            }
            set(grayScale.asVector(), getWidth(), getHeight(), alphaChannel ? 2 : 1);
            return;
         }
      }
      else {
         // We're currently RGB or RGBA and need to be Grayscale or Grayscale + Alpha
         // RGB weights from <https://en.wikipedia.org/wiki/Grayscale>, citing Pratt, Digital Image Processing
         const float rgbWeights[3] = { mRToGray, mGToGray, mBToGray };//{0.30f, 0.59f, 0.11f};
         Buffer grayScale(getWidth(), getHeight(), alphaChannel ? 2 : 1);

         for (int y = 0; y < getHeight(); ++y) {
            for (int x = 0; x < getWidth(); ++x) {
               float sum = 0.0f;
               for (int f = 0; f < 3; ++f) {
                  sum += at(x, y, f) * rgbWeights[f];
               }
               grayScale.set(x, y, 0, sum);
               if (alphaChannel) {
                  if (getFeatures() > 3) {
                     grayScale.set(x, y, 1, at(x, y, 3));
                  }
                  else {
                     grayScale.set(x, y, 1, 1.0f);
                  }
               }
            }
         }
         set(grayScale.asVector(), getWidth(), getHeight(), alphaChannel ? 2 : 1);
      }
   }

   void Image::convertToColor(bool alphaChannel) {
      // Are we already color? If so, do we need to add or remove an alpha channel?
      if (getFeatures() > 2) {
         if ((getFeatures() == 3 && !alphaChannel) || (getFeatures() == 4 && alphaChannel)) {
            // This is the correct format already, nothing to be done
            return;
         }
         else {
            // We're already color, but we're adding or removing an alpha channel
            Buffer color(getWidth(), getHeight(), alphaChannel ? 4 : 3);
            for (int y = 0; y < getHeight(); ++y) {
               for (int x = 0; x < getWidth(); ++x) {
                  color.set(x, y, mRPos, at(x, y, mRPos));
                  color.set(x, y, mGPos, at(x, y, mGPos));
                  color.set(x, y, mBPos, at(x, y, mBPos));
                  if (alphaChannel) { 
                     color.set(x, y, mAPos, 1.0f);
                  }
               }
            }
            set(color.asVector(), getWidth(), getHeight(), alphaChannel ? 4 : 3);
         }
      }
      else {
         // We're converting a grayscale image to color
         Buffer color(getWidth(), getHeight(), alphaChannel ? 4 : 3);
         for (int y = 0; y < getHeight(); ++y) {
            for (int x = 0; x < getWidth(); ++x) {
               float val = at(x, y, 0);
               color.set(x, y, mRPos, val);
               color.set(x, y, mGPos, val);
               color.set(x, y, mBPos, val);
               if (alphaChannel) { 
                  if (getFeatures() == 2) {
                     color.set(x, y, mAPos, at(x, y, 1));
                  }
                  else {
                     color.set(x, y, mAPos, 1.0f);
                  }
               }
            }
         }
         set(color.asVector(), getWidth(), getHeight(), alphaChannel ? 4 : 3);
      }
   }

   void Image::read(std::string filename) {
      int width = 0, height = 0, channels = 0;
      uint8_t* data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
      pvErrorIf(data == nullptr, " File not found: %s\n", filename.c_str());
      resize(height, width, channels);

      for (int y = 0; y < height; ++y) {
         for (int x = 0; x < width; ++x) {
            for (int f = 0; f < channels; ++f) {
               float value = static_cast<float>(data[(y * width + x) * channels + f]) / 255.0f;
               set(x, y, f, value);
            }
         }
      }
      
      stbi_image_free(data);
   }
   
   void Image::write(std::string filename) {
      std::vector<uint8_t> byteData(getWidth() * getHeight() * getFeatures());
      int byteIndex = 0;
      float imageMin = 0.0f;
      float imageMax = 1.0f;

      for (int y = 0; y < getHeight(); ++y) {
         for (int x = 0; x < getWidth(); ++x) {
            for (int f = 0; f < getFeatures(); ++f) {
               imageMin = at(x, y, f) < imageMin ? at(x, y, f) : imageMin;
               imageMax = at(x, y, f) > imageMax ? at(x, y, f) : imageMax;
            }
         }
      }

      for (int y = 0; y < getHeight(); ++y) {
         for (int x = 0; x < getWidth(); ++x) {
            for (int f = 0; f < getFeatures(); ++f) {
               float normVal = (at(x, y, f) - imageMin) / (imageMax - imageMin);
               byteData.at(byteIndex++) = static_cast<uint8_t>(normVal * 255.0f);
            }
         }
      }

      stbi_write_png(filename.c_str(), getWidth(), getHeight(), getFeatures(), byteData.data(), getWidth() * getFeatures());
   }
}
