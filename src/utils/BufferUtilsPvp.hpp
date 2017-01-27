#ifndef __BUFFERUTILSPVP_HPP__
#define __BUFFERUTILSPVP_HPP__

#include "io/FileStream.hpp"
#include "structures/Buffer.hpp"
#include "structures/SparseList.hpp"

#include <string>
#include <vector>

using std::vector;
using std::string;

namespace PV {

namespace BufferUtils {

/**
 * The enum for the dataType field of the pvp file header.
 */
typedef enum HeaderDataTypeEnum {
   // Values are hardcoded to ensure consistency between builds.
   UNRECOGNIZED_DATATYPE = 0,
   BYTE                  = 1,
   INT                   = 2,
   FLOAT                 = 3,
   // datatype 4 is obsolete;
   TAUS_UINT4 = 5,
} HeaderDataType;

// This structure is used to avoid having to traverse
// a sparse pvp file from start to finish every time
// we want to load data from it.
struct SparseFileTable {
   vector<long> frameStartOffsets;
   vector<long> frameLengths;
   bool valuesIncluded;
};

struct ActivityHeader {
   int headerSize, numParams, fileType, nx, ny, nf, numRecords, recordSize, dataSize, dataType,
         nxProcs, nyProcs, nxGlobal, nyGlobal, kx0, ky0, nBatch, nBands;
   double timestamp;
};

template <typename T>
void writeFrame(FileStream &fStream, Buffer<T> *buffer, double timeStamp);

template <typename T>
double readFrame(FileStream &fStream, Buffer<T> *buffer);

template <typename T>
BufferUtils::HeaderDataType returnDataType();

template <typename T>
struct ActivityHeader buildActivityHeader(int width, int height, int features, int numFrames);

template <typename T>
struct ActivityHeader buildSparseActivityHeader(int width, int height, int features, int numFrames);

template <typename T>
void writeToPvp(const char *fName, Buffer<T> *buffer, double timeStamp, bool verifyWrites = false);

template <typename T>
void appendToPvp(
      const char *fName,
      Buffer<T> *buffer,
      int frameWriteIndex,
      double timeStamp,
      bool verifyWrites = false);

/**
 * Reads a frame from an activity layer of any activity file type into a buffer.
 * The buffer will be resized to the size indicated in the pvp file's header.
 */
template <typename T>
double readActivityFromPvp(char const *fName, Buffer<T> *buffer, int frameReadIndex);

/**
 * Reads a frame from a nonspiking activity layer into a buffer. If the file type
 * is anything else, exits with an error.
 */
template <typename T>
double readDenseFromPvp(const char *fName, Buffer<T> *buffer, int frameReadIndex);

template <typename T>
void writeSparseFrame(FileStream &fStream, SparseList<T> *list, double timeStamp);

template <typename T>
double readSparseFrame(FileStream &fStream, SparseList<T> *list);

template <typename T>
double readSparseBinaryFrame(FileStream &fStream, SparseList<T> *list, T oneVal);

template <typename T>
void writeSparseToPvp(
      const char *fName,
      SparseList<T> *list,
      double timeStamp,
      int width,
      int height,
      int features,
      bool verifyWrites = false);

template <typename T>
void appendSparseToPvp(
      const char *fName,
      SparseList<T> *list,
      int frameWriteIndex,
      double timeStamp,
      bool verifyWrites = false);

template <typename T>
double readSparseFromPvp(
      const char *fName,
      SparseList<T> *list,
      int frameReadIndex,
      SparseFileTable *cachedTable = nullptr);

/**
 * Reads a frame from a sparse-values pvp file into a (nonsparse) buffer.
 * Neither the list of active indices nor the SparseFileTable is returned.
 * Use the readSparseFromPvp function to get the SparseList or SparseFileTable.
 */
template <typename T>
double readDenseFromSparsePvp(char const *fName, Buffer<T> *buffer, int frameReadIndex);

template <typename T>
double readSparseBinaryFromPvp(
      const char *fName,
      SparseList<T> *list,
      int frameReadIndex,
      T oneVal,
      SparseFileTable *cachedTable = nullptr);

/**
 * Reads a frame from a sparse-binary pvp file into a (nonsparse) buffer.
 * Neither the list of active indices nor the SparseFileTable is returned.
 * Use the readSparseBinaryFromPvp function to get the SparseList or SparseFileTable.
 */
template <typename T>
double readDenseFromSparseBinaryPvp(char const *fName, Buffer<T> *buffer, int frameReadIndex);

static void writeActivityHeader(FileStream &fStream, struct ActivityHeader const &header);
static struct ActivityHeader readActivityHeader(FileStream &fStream);
static SparseFileTable buildSparseFileTable(FileStream &fStream, int upToIndex);
}
}

#include "BufferUtilsPvp.tpp"

#endif
