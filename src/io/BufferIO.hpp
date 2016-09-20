#ifndef __BUFFERIO_HPP__
#define __BUFFERIO_HPP__

#include "FileStream.hpp"
#include "utils/Buffer.hpp"

#include <vector>
#include <string>

using std::vector;
using std::string;

namespace PV {

class BufferIO {
   public:
      template <typename T>
      static void writeFrame(FileStream *fStream,
                             Buffer<T> *buffer,
                             double timeStamp);
      template <typename T>
      static double readFrame(FileStream *fStream,
                              Buffer<T> *buffer,
                              int frameReadIndex);
      template <typename T>
      static vector<int> buildHeader(int width,
                                     int height,
                                     int features,
                                     int numFrames);
      template <typename T>
      static void writeToPvp(string fName,
                             Buffer<T> *buffer,
                             double timeStamp);
      template <typename T>
      static void appendToPvp(string fName,
                              Buffer<T> *buffer,
                              int frameWriteIndex,
                              double timeStamp);
      static void writeHeader(FileStream *fStream, vector<int> header);
      static vector<int> readHeader(FileStream *fStream);
};
}

#include "BufferIO.tpp"

#endif
