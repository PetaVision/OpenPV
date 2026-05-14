#include "cMakeHeader.h"
#include "Image.hpp"
#include "Buffer.hpp"
#include "include/pv_common.h"
#include "utils/PVLog.hpp"

// These defines are required by the stb headers
#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "io/stb_image.h"
#endif
#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "io/stb_image_write.h"
#endif

#include <cstring>
#include <fstream>

#ifdef PV_USE_TIFF
#include <cinttypes>
#include <tiffio.h>
#endif // PV_USE_TIFF

namespace PV {

Image::Image(std::string filename) { read(filename); }

Image::Image(const std::vector<float> &data, int width, int height, int channels) {
   set(data, width, height, channels);
}

Image::Image() : Buffer<float>() {}

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

void Image::convertToGray(bool alphaChannelFlag) {
   if (getFeatures() < 3) {
      if ((getFeatures() == 1 && !alphaChannelFlag) || (getFeatures() == 2 && alphaChannelFlag)) {
         // Do nothing if we are already in the correct format
         return;
      }
      else {
         // We are already grayscale, but we're adding or removing an alpha channel
         Buffer<float> grayScale(getWidth(), getHeight(), alphaChannelFlag ? 2 : 1);
         for (int y = 0; y < getHeight(); ++y) {
            for (int x = 0; x < getWidth(); ++x) {
               grayScale.set(x, y, 0, at(x, y, 0));
               if (alphaChannelFlag) {
                  grayScale.set(x, y, 1, 1.0f);
               }
            }
         }
         set(grayScale.asVector(), getWidth(), getHeight(), alphaChannelFlag ? 2 : 1);
         return;
      }
   }
   else {
      // We're currently RGB or RGBA and need to be Grayscale or Grayscale + Alpha
      // RGB weights from <https://en.wikipedia.org/wiki/Grayscale>, citing Pratt, Digital Image
      // Processing
      const float rgbWeights[3] = {mRToGray, mGToGray, mBToGray}; //{0.30f, 0.59f, 0.11f};
      Buffer<float> grayScale(getWidth(), getHeight(), alphaChannelFlag ? 2 : 1);

      for (int y = 0; y < getHeight(); ++y) {
         for (int x = 0; x < getWidth(); ++x) {
            float sum = 0.0f;
            for (int f = 0; f < 3; ++f) {
               sum += at(x, y, f) * rgbWeights[f];
            }
            grayScale.set(x, y, 0, sum);
            if (alphaChannelFlag) {
               if (getFeatures() > 3) {
                  grayScale.set(x, y, 1, at(x, y, 3));
               }
               else {
                  grayScale.set(x, y, 1, 1.0f);
               }
            }
         }
      }
      set(grayScale.asVector(), getWidth(), getHeight(), alphaChannelFlag ? 2 : 1);
   }
}

void Image::convertToColor(bool alphaChannelFlag) {
   // Are we already color? If so, do we need to add or remove an alpha channel?
   if (getFeatures() > 2) {
      if ((getFeatures() == 3 && !alphaChannelFlag) || (getFeatures() == 4 && alphaChannelFlag)) {
         // This is the correct format already, nothing to be done
         return;
      }
      else {
         // We're already color, but we're adding or removing an alpha channel
         Buffer<float> color(getWidth(), getHeight(), alphaChannelFlag ? 4 : 3);
         for (int y = 0; y < getHeight(); ++y) {
            for (int x = 0; x < getWidth(); ++x) {
               color.set(x, y, mRPos, at(x, y, mRPos));
               color.set(x, y, mGPos, at(x, y, mGPos));
               color.set(x, y, mBPos, at(x, y, mBPos));
               if (alphaChannelFlag) {
                  color.set(x, y, mAPos, 1.0f);
               }
            }
         }
         set(color.asVector(), getWidth(), getHeight(), alphaChannelFlag ? 4 : 3);
      }
   }
   else {
      // We're converting a grayscale image to color
      Buffer<float> color(getWidth(), getHeight(), alphaChannelFlag ? 4 : 3);
      for (int y = 0; y < getHeight(); ++y) {
         for (int x = 0; x < getWidth(); ++x) {
            float val = at(x, y, 0);
            color.set(x, y, mRPos, val);
            color.set(x, y, mGPos, val);
            color.set(x, y, mBPos, val);
            if (alphaChannelFlag) {
               if (getFeatures() == 2) {
                  color.set(x, y, mAPos, at(x, y, 1));
               }
               else {
                  color.set(x, y, mAPos, 1.0f);
               }
            }
         }
      }
      set(color.asVector(), getWidth(), getHeight(), alphaChannelFlag ? 4 : 3);
   }
}

void Image::read(std::string const &filename) {
   // Test if file is a TIFF
   std::ifstream filestream(filename);
   FatalIf(!filestream, "Image::read() Unable to open %s\n", filename.c_str());
   char fh[4];
   filestream.read(fh, 4);
   FatalIf(!filestream, "Unable to read %s\n", filename.c_str());
   filestream.close();
   // Test if fileheader corresponds to TIFF
   bool tiffLittleEndian = (fh[0] == 0x49 and fh[1] == 0x49 and fh[2] == 0x2a and fh[3] == 0x00);
   bool tiffBigEndian = (fh[0] == 0x4d and fh[1] == 0x4d and fh[2] == 0x00 and fh[3] == 0x2a);
   if (tiffBigEndian or tiffLittleEndian) {
      readTIFF(filename);
   }
   else {
      // Try stb_image
      readSTB(filename);
   }
}

void Image::readSTB(std::string const &filename) {
   int width = 0, height = 0, channels = 0;
   stbi_us *data = stbi_load_16(filename.c_str(), &width, &height, &channels, 0);
   if (data == nullptr) {
      if (!std::strcmp(stbi_failure_reason(), "unknown image type")) {
         Fatal().printf(
               " File \"%s\" is an unknown image type.\n"
               " (A list of image files must have a .txt extension;"
               " an individual image file must be readable by the stb_image library.)\n",
               filename.c_str());
      }
      else {
         Fatal().printf("Unable to load \"%s\": %s.\n", filename.c_str(), stbi_failure_reason());
      }
   }
   FatalIf(data == nullptr, " File not found: %s\n", filename.c_str());
   resize(width, height, channels);

   for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
         for (int f = 0; f < channels; ++f) {
            float value = static_cast<float>(data[(y * width + x) * channels + f]) / 65535.0f;
            set(x, y, f, value);
         }
      }
   }

   stbi_image_free(data);
}

void Image::readTIFF(std::string const &filename) {
#ifdef PV_USE_TIFF
   // Load a tiff file
   TIFF *tiff = TIFFOpen(filename.c_str(), "r");
   FatalIf(tiff == nullptr, "Unable to open TIFF \"%s\"\n", filename.c_str());
   std::uint32_t width, height;
   std::uint16_t features;
   int status = PV_SUCCESS;
   if (TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width) == 0) {
      ErrorLog().printf("Unable to read image width of \"%s\"\n", filename.c_str());
      status = PV_FAILURE;
   };
   if (TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height) == 0) {
      ErrorLog().printf("Unable to read image height of \"%s\"\n", filename.c_str());
      status = PV_FAILURE;
   };
   if (TIFFGetField(tiff, TIFFTAG_SAMPLESPERPIXEL, &features) == 0) {
      ErrorLog().printf("Unable to read image height of \"%s\"\n", filename.c_str());
      status = PV_FAILURE;
   };
   if (status != PV_SUCCESS) {
      exit(EXIT_FAILURE);
   }
   resize(width, height, features);
   int h = static_cast<int>(height);
   int w = static_cast<int>(width);
   int area = width * height;
   std::vector<uint32_t> raster(area);
   status = TIFFReadRGBAImage(tiff, width, height, raster.data(), 0);
   status = status ? PV_SUCCESS : PV_FAILURE; // In libtiff, 1 is success and 0 is an error
   FatalIf(
         status != PV_SUCCESS,
         "TIFFReadRGBAImage() was unable to read \"%s\".\n",
         filename.c_str());
   TIFFClose(tiff);
   switch (features) {
      case 1:
         for (int k = 0; k < area; ++k) {
            int y = h - 1 - (k / w);
            int x = k % w;
            uint32_t v = raster[k];
            float vf = static_cast<float>(TIFFGetR(v)) / 255.0f;
            set(x, y, 0, vf);
         }
         break;
      case 3:
         for (int k = 0; k < area; ++k) {
            int y = h - 1 - (k / w);
            int x = k % w;
            uint32_t v = raster[k];
            set(x, y, 0, static_cast<float>(TIFFGetR(v)) / 255.0f);
            set(x, y, 1, static_cast<float>(TIFFGetG(v)) / 255.0f);
            set(x, y, 2, static_cast<float>(TIFFGetB(v)) / 255.0f);
         }
         break;
      case 4:
         for (int k = 0; k < area; ++k) {
            int y = h - 1 - (k / w);
            int x = k % w;
            uint32_t v = raster[k];
            set(x, y, 0, static_cast<float>(TIFFGetR(v)) / 255.0f);
            set(x, y, 1, static_cast<float>(TIFFGetG(v)) / 255.0f);
            set(x, y, 2, static_cast<float>(TIFFGetB(v)) / 255.0f);
            set(x, y, 3, static_cast<float>(TIFFGetA(v)) / 255.0f);
         }
         break;
      default:
         Fatal().printf(
               "Currently unable to read TIFF \"%s\" with %" PRIu16 " samples per pixel\n",
               filename.c_str(), features);
         break;
   }
#else // PV_USE_TIFF
   Fatal().printf(
         "PetaVision was compiled with the PV_USE_TIFF option off; unable to read TIFF \"%s\".\n",
         filename.c_str());
#endif // PV_USE_TIFF

}

void Image::write(std::string const &filename) {
   std::vector<uint16_t> byteData(getWidth() * getHeight() * getFeatures());
   int byteIndex  = 0;
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
            float normVal            = (at(x, y, f) - imageMin) / (imageMax - imageMin);
            byteData.at(byteIndex++) = static_cast<uint16_t>(normVal * 65535.0f);
         }
      }
   }

   stbi_write_png(
         filename.c_str(),
         getWidth(),
         getHeight(),
         getFeatures(),
         byteData.data(),
         getWidth() * getFeatures());
}
} // namespace PV
