#ifndef __BUFFERUTILSPVP_HPP__
#define __BUFFERUTILSPVP_HPP__

#include "io/FileStream.hpp"
#include "structures/Buffer.hpp"
#include "structures/SparseList.hpp"

#include <vector>
#include <string>

using std::vector;
using std::string;

namespace PV {

   namespace BufferUtils {

      // This structure is used to avoid having to traverse
      // a sparse pvp file from start to finish every time
      // we want to load data from it.
      struct SparseFileTable {
        vector<long> frameStartOffsets;
        vector<int>  frameLengths;
        bool         valuesIncluded;
      }; 

      template <typename T>
      static void writeFrame(FileStream *fStream,
                             Buffer<T> *buffer,
                             double timeStamp);
      template <typename T>
      static double readFrame(FileStream *fStream,
                              Buffer<T> *buffer);
      template <typename T>
      static vector<int> buildHeader(int width,
                                     int height,
                                     int features,
                                     int numFrames);
      template <typename T>
      static void writeToPvp(const char * fName,
                             Buffer<T> *buffer,
                             double timeStamp);
      template <typename T>
      static void appendToPvp(const char *fName,
                              Buffer<T> *buffer,
                              int frameWriteIndex,
                              double timeStamp);
      template <typename T>
      static double readFromPvp(const char *fName,
                                Buffer<T> *buffer,
                                int frameReadIndex);
      template <typename T>
      static void writeSparseFrame(FileStream *fStream,
                                   SparseList<T> *list,
                                   double timeStamp);
      template <typename T>
      static double readSparseFrame(FileStream *fStream,
                                    SparseList<T> *list);
      template <typename T>
      static void writeSparseToPvp(const char *fName,
                                   SparseList<T> *list,
                                   double timeStamp,
                                   int width,
                                   int height,
                                   int features);
      template <typename T>
      static void appendSparseToPvp(const char *fName,
                                   SparseList<T> *list,
                                   int frameWriteIndex,
                                   double timeStamp);
      template <typename T>
      static double readSparseFromPvp(const char *fName,
                                      SparseList<T> *list,
                                      int frameReadIndex);
      
      static void writeHeader(FileStream *fStream, vector<int> header);
      static vector<int> readHeader(FileStream *fStream);

      static SparseFileTable buildSparseFileTable(FileStream *fStream,
                                                  int upToIndex);
   }
}

#include "BufferUtilsPvp.tpp"

#endif
