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

static void writeActivityHeader(FileStream &fStream, ActivityHeader const &header);
static ActivityHeader readActivityHeader(FileStream &fStream);
static SparseFileTable buildSparseFileTable(FileStream &fStream, int upToIndex);

template <typename T>
std::size_t weightPatchSize(int numWeightsInPatch);

std::size_t weightPatchSize(int numWeightsInPatch, bool compressed);

/**
 * Builds a header for weight files, either shared or nonshared, with
 * minimal processing of arguments.
 *
 * The recordSize field is computed from nxp, nyp, nfp, numPatches and
 * the compress flag. The fileType field is determined by the shared flag.
 * Other header fields are copied from the appropriate argument.
 * (For example, the preLayerNx argument is placed into the nx and nxExtended
 * fields).
 *
 * This function is called by both the buildSharedWeightHeader and
 * buildNonsharedWeightHeader functions.
 */
WeightHeader buildWeightHeader(
      bool shared,
      int preLayerNx,
      int preLayerNy,
      int preLayerNf,
      int preLayerNxExt,
      int preLayerNyExt,
      int numArbors,
      double timestamp,
      int nxp,
      int nyp,
      int nfp,
      bool compress,
      float minVal,
      float maxVal);

/**
 * Builds a header for shared-weight files.
 * nxp, nyp, nfp are the dimensions of one patch.
 * numArbors is the number of arbors.
 * numPatchesX, numPatchesY, numPatchesF are the dimensions of the array of
 * patches in one process.
 *
 * timestamp is value for the timestamp header field.
 * preLayerLoc is the presynaptic layer's PVLayerLoc.
 * numProcessesInColumn and numProcessesInRow are the dimensions of
 * the MPIBlock being used with this header. These arguments are used to
 * modify the nx, ny, nxGlobal, and nyGlobal header fields.
 *
 * minVal and maxVal are the values of the corresponding fields of
 * the header.
 *
 * The compress flag indicates whether the weights will be byte or float
 * values. It affects the datatype, datasize, and recordSize fields.
 */
WeightHeader buildSharedWeightHeader(
      int nxp,
      int nyp,
      int nfp,
      int numArbors,
      int numPatchesX,
      int numPatchesY,
      int numPatchesF,
      double timestamp,
      bool compress,
      float minVal,
      float maxVal);

/**
 * Builds a header for nonshared-weight files.
 * nxp, nyp, nfp are the dimensions of one patch.
 * numArbors is the number of arbors.
 * If the extended flag is on, there is one patch for each presynaptic
 * neuron in extended space (as defined by the preLayerLoc argument).
 * Otherwise, there is one patch for each presynaptic neuron in the restricted
 * space only.
 *
 * timestamp is value for the timestamp header field.
 * preLayerLoc is the presynaptic layer's PVLayerLoc.
 * numProcessesInColumn and numProcessesInRow are the dimensions of
 * the MPIBlock being used with this header. These arguments are used to
 * modify the nx, ny, nxGlobal, and nyGlobal header fields.
 *
 * minVal and maxVal are the values of the corresponding fields of
 * the header.
 *
 * The compress flag indicates whether the weights will be byte or float
 * values. It affects the datatype, datasize, and recordSize fields.
 */
WeightHeader buildNonsharedWeightHeader(
      int nxp,
      int nyp,
      int nfp,
      int numArbors,
      bool extended,
      double timestamp,
      PVLayerLoc const *preLayerLoc,
      PVLayerLoc const *postLayerLoc,
      int numColumnProcesses,
      int numRowProcesses,
      float minVal,
      float maxVal,
      bool compress);

void calcNumberOfPatches(
      PVLayerLoc const *preLayerLoc,
      PVLayerLoc const *postLayerLoc,
      int numColumnProcesses,
      int numRowProcesses,
      bool extended,
      int nxp,
      int nyp,
      int &numPatchesX,
      int &numPatchesY,
      int &numPatchesF,
      int &numPatchesXExt,
      int &numPatchesYExt);
} // end namespace BufferUtils
} // end namespace PV

#include "BufferUtilsPvp.tpp"

#endif
