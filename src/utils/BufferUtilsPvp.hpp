#ifndef BUFFERUTILSPVP_HPP_
#define BUFFERUTILSPVP_HPP_

#include "include/PVLayerLoc.hpp"
#include "io/FileStream.hpp"
#include "structures/Buffer.hpp"
#include "structures/SparseList.hpp"

#include <cstdlib>
#include <vector>

// File type of activities where there are no timestamps in the individual frames.  No longer used
#define PVP_FILE_TYPE 1

// File type for binary layers (activity is sparse and values are only 1 or 0). No longer used
#define PVP_ACT_FILE_TYPE 2

// File type for connections without shared weights
#define PVP_WGT_FILE_TYPE 3

// File type for nonsparse layers and checkpoint files for all layers
#define PVP_NONSPIKING_ACT_FILE_TYPE 4

// File type for connections with shared weights
#define PVP_KERNEL_FILE_TYPE 5

// File type for sparse layers. The locations and values of nonzero neurons are stored.
#define PVP_ACT_SPARSEVALUES_FILE_TYPE 6

#define NUM_BIN_PARAMS (18 + sizeof(double) / sizeof(int))

#define NUM_WGT_EXTRA_PARAMS 6
#define NUM_WGT_PARAMS (NUM_BIN_PARAMS + NUM_WGT_EXTRA_PARAMS)

#define INDEX_HEADER_SIZE 0
#define INDEX_NUM_PARAMS 1
#define INDEX_FILE_TYPE 2
#define INDEX_NX 3
#define INDEX_NY 4
#define INDEX_NF 5
#define INDEX_NUM_RECORDS 6
#define INDEX_RECORD_SIZE 7
#define INDEX_DATA_SIZE 8
#define INDEX_DATA_TYPE 9
#define INDEX_NX_PROCS 10
#define INDEX_NY_PROCS 11
#define INDEX_NX_EXTENDED 12
#define INDEX_NY_EXTENDED 13
#define INDEX_KX0 14
#define INDEX_KY0 15
#define INDEX_NBATCH 16
#define INDEX_NBANDS 17
#define INDEX_TIME 18

// these are extra parameters used by weight files
//
#define INDEX_WGT_NXP 0
#define INDEX_WGT_NYP 1
#define INDEX_WGT_NFP 2
#define INDEX_WGT_MIN 3
#define INDEX_WGT_MAX 4
#define INDEX_WGT_NUMPATCHES 5
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
   std::vector<long> frameStartOffsets;
   std::vector<long> frameLengths;
};

struct ActivityHeader {
   int headerSize, numParams, fileType, nx, ny, nf, numRecords, recordSize, dataSize, dataType,
         nxProcs, nyProcs, nxExtended, nyExtended, kx0, ky0, nBatch, nBands;
   double timestamp;
};

struct WeightHeader {
   ActivityHeader baseHeader;
   int nxp, nyp, nfp;
   float minVal, maxVal;
   int numPatches;
};

template <typename T>
void writeFrame(FileStream &fStream, Buffer<T> *buffer, double timeStamp);

template <typename T>
double readFrame(FileStream &fStream, Buffer<T> *buffer);

template <typename T>
double readFrameWindow(
      FileStream &fStream,
      Buffer<T> *buffer,
      ActivityHeader const &header,
      int xStart,
      int yStart,
      int fStart);

template <typename T>
BufferUtils::HeaderDataType returnDataType();

template <typename T>
ActivityHeader buildActivityHeader(int width, int height, int features, int numFrames);

template <typename T>
ActivityHeader buildSparseActivityHeader(int width, int height, int features, int numFrames);

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
 * If the SparseFileTable pointer is null, it is ignored. If it is not null and
 * the path points to a sparse-binary or sparse-values activity file, the table
 * is used to speed navigation of the pvp file. If the SparseFileTable is empty
 * it is initialized.
 */
template <typename T>
double readActivityFromPvp(
      char const *fName,
      Buffer<T> *buffer,
      int frameReadIndex,
      BufferUtils::SparseFileTable *const sparseFileTable);

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
double readDenseFromSparsePvp(
      char const *fName,
      Buffer<T> *buffer,
      int frameReadIndex,
      SparseFileTable *sparseFileTable);

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
double readDenseFromSparseBinaryPvp(
      char const *fName,
      Buffer<T> *buffer,
      int frameReadIndex,
      SparseFileTable *sparseFileTable);

void writeActivityHeader(FileStream &fStream, ActivityHeader const &header);

/**
 * Reads a pvp header and returns it in vector format. Leaves inStream
 * pointing at the start of the first frame.
 */
ActivityHeader readActivityHeader(FileStream &fStream);

/**
 * Builds a table of offsets and lengths for each pvp frame
 * index up to (but not including) upToIndex. Works for both
 * sparse activity and sparse binary files. Leaves the input
 * stream pointing at the location where frame upToIndex would
 * begin.
 */
SparseFileTable buildSparseFileTable(FileStream &fStream, int upToIndex);

template <typename T>
std::size_t weightPatchSize(int numWeightsInPatch);

std::size_t weightPatchSize(int numWeightsInPatch, bool compressed);

} // end namespace BufferUtils

} // end namespace PV

#include "BufferUtilsPvp.tpp"

#endif // BUFFERUTILSPVP_HPP_
