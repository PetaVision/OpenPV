#include "PVImg.hpp"
#include "PVLog.hpp"

namespace PV {

   PVImg::PVImg(std::string filename) {
      int channels = 0;
      // Passing 4 ensures our data structure is padded for 4 channels,
      // even if that data was not present in the file. &c is filled
      // with the actual number of channels in the file.
      uint8_t* data = stbi_load(filename.c_str(), &mWidth, &mHeight, &channels, 4);
      if(data == nullptr) {
         pvError() << " File not found: " << filename << std::endl;
      }

      mData.resize(mHeight);
      for(int row = 0; row < mHeight; ++row) {
         mData.at(row).resize(mWidth);
         for(int col = 0; col < mWidth; ++col) {
            float r = static_cast<float>(*(data + row * mWidth * 4 + col * 4 + mRPos)) / 255.0f;
            float g = static_cast<float>(*(data + row * mWidth * 4 + col * 4 + mGPos)) / 255.0f;
            float b = static_cast<float>(*(data + row * mWidth * 4 + col * 4 + mBPos)) / 255.0f;
            float a = static_cast<float>(*(data + row * mWidth * 4 + col * 4 + mAPos)) / 255.0f;
            
            mData.at(row).at(col).resize(4);

            switch(channels) {
               case 1: // Greyscale
                  mData.at(row).at(col).at(mRPos) = r;
                  mData.at(row).at(col).at(mGPos) = r;
                  mData.at(row).at(col).at(mBPos) = r;
                  mData.at(row).at(col).at(mAPos) = 1.0f;
                  break;
               case 2: // Greyscale with Alpha
                  mData.at(row).at(col).at(mRPos) = r;
                  mData.at(row).at(col).at(mGPos) = r;
                  mData.at(row).at(col).at(mBPos) = r;
                  mData.at(row).at(col).at(mAPos) = a;
               case 3: // RGB
                  mData.at(row).at(col).at(mRPos) = r;
                  mData.at(row).at(col).at(mGPos) = g;
                  mData.at(row).at(col).at(mBPos) = b;
                  mData.at(row).at(col).at(mAPos) = 1.0f;
               case 4: // RGBA
                  mData.at(row).at(col).at(mRPos) = r;
                  mData.at(row).at(col).at(mGPos) = g;
                  mData.at(row).at(col).at(mBPos) = b;
                  mData.at(row).at(col).at(mAPos) = a;
               default:
                  pvError() << " Invalid color format for file: " << filename << std::endl;
                  break;
            }
         }
      }
      
      stbi_image_free(data);
   }

   PVImg::PVImg(const std::vector<float> &data, int width, int height, int channels) {
      deserialize(data, width, height, channels);
   }

   void PVImg::setPixel(int x, int y, float r, float g, float b) {
      setPixel(x, y, r, g, b, 1.0f);
   }

   void PVImg::setPixel(int x, int y, float r, float g, float b, float a) {
      mData.at(y).at(x).at(0) = r;
      mData.at(y).at(x).at(1) = g;
      mData.at(y).at(x).at(2) = b;
      mData.at(y).at(x).at(3) = a;
   }

   std::vector<float> PVImg::serialize(int channels) {

      std::vector<int> channelIndices;

      switch(channels) {
         case 1:
            channelIndices = { 0 };
            break;
         case 2:
            channelIndices = { 0, 3 };
            break;
         case 3:
            channelIndices = { 0, 1, 2 };
            break;
         case 4:
            channelIndices = { 0, 1, 2, 3 };
            break;
         default:
            pvError() << "Invalid argument: Channels must be between 1 and 4 inclusive." << std::endl;
            break;
      }

      std::vector<float> result(mWidth * mHeight * channels);
      for(int y = 0; y < mHeight; ++y) {
         for(int x = 0; x < mWidth; ++x) {
            for(int c = 0; c < channels; ++c) {
               result[channels * (y * mWidth + x) + c] = mData.at(y).at(x).at(channelIndices.at(c));
            }
         }
      }
      return result;
   }

   void PVImg::deserialize(const std::vector<float> &data, int width, int height, int channels) {
      mHeight = height;
      mWidth = width;
      std::vector<int> channelIndices;

      switch(channels) {
         case 1:
            channelIndices = { 0, 0, 0 };
            break;
         case 2:
            channelIndices = { 0, 0, 0, 1 };
            break;
         case 3:
            channelIndices = { 0, 1, 2 };
            break;
         case 4:
            channelIndices = { 0, 1, 2, 3 };
            break;
         default:
            pvError() << "Invalid argument: Channels must be between 1 and 4 inclusive." << std::endl;
            break;
      }

      mData.clear();
      mData.resize(mHeight);
      for(int y = 0; y < mHeight; ++y) {
         mData.at(y).resize(mWidth);
         for(int x = 0; x < mWidth; ++x) {
            mData.at(y).at(x).resize(4);
            mData.at(y).at(x).at(3) = 1.0f; //Default to alpha = 1.0
            for(int c = 0; c < channelIndices.size(); ++c) {
               mData.at(y).at(x).at(c) = data[channels * (y * mWidth + x) + channelIndices.at(c)];
            }
         }
      }
   }
   
}
