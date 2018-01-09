#include "io/io.hpp"
#include "utils/conversions.h"
#include <limits>

namespace PV {

namespace BufferUtils {

// Write a single frame to a pvp file, starting at fStream's location.
// A pvp file may contain multiple frames.
template <typename T>
void writeFrame(FileStream &fStream, Buffer<T> *buffer, double timestamp) {
   size_t dataSize = sizeof(T);
   fStream.write(&timestamp, sizeof(double));
   fStream.write(buffer->asVector().data(), buffer->getTotalElements() * dataSize);
}

// Reads the next frame from a pvp file. Returns the timeStamp.
// Assumes that buffer is already the correct dimensions for
// the expected data.
template <typename T>
double readFrame(FileStream &fStream, Buffer<T> *buffer) {
   double timeStamp;
   fStream.read(&timeStamp, sizeof(double));

   vector<T> data(buffer->getTotalElements());
   size_t expected = data.size() * sizeof(T);
   fStream.read(data.data(), expected);

   buffer->set(data, buffer->getWidth(), buffer->getHeight(), buffer->getFeatures());
   return timeStamp;
}

template <typename T>
double readFrameWindow(
      FileStream &fStream,
      Buffer<T> *buffer,
      ActivityHeader const &header,
      int xStart,
      int yStart,
      int fStart) {
   double timeStamp;
   fStream.read(&timeStamp, sizeof(double));

   long frameDataStart = fStream.getOutPos();

   int const nx = buffer->getWidth();
   int const ny = buffer->getHeight();
   int const nf = buffer->getFeatures();

   int const nxGlobal = header.nx;
   int const nyGlobal = header.ny;
   int const nfGlobal = header.nf;

   FatalIf(xStart < 0, "readFrameWindow: window's left edge %d is negative.\n", xStart);
   FatalIf(yStart < 0, "readFrameWindow: window's top edge %d is negative.\n", yStart);
   FatalIf(fStart < 0, "readFrameWindow: starting feature %d cannot be negative.\n", fStart);

   FatalIf(nx <= 0, "readFrameWindow: window's width %d must be positive.\n", nx);
   FatalIf(ny <= 0, "readFrameWindow: window's height %d must be positive.\n", nx);
   FatalIf(nf <= 0, "readFrameWindow: window's number of features %d must be positive.\n", nx);

   FatalIf(
         xStart + nx > nxGlobal,
         "readFrameWindow: window's right edge %d exceeds file's width %d.\n",
         xStart + nx,
         nxGlobal);
   FatalIf(
         yStart + ny > nyGlobal,
         "readFrameWindow: window's bottom edge %d exceeds file's width %d.\n",
         yStart + ny,
         nyGlobal);
   FatalIf(
         fStart + nf > nfGlobal,
         "readFrameWindow: window's last feature %d exceeds file's number of features %d.\n",
         fStart + nf,
         nfGlobal);

   vector<T> bufferData(nx * ny * nf);
   if (nfGlobal == buffer->getFeatures()) {
      std::size_t lineWidth = sizeof(T) * (std::size_t)(nfGlobal * buffer->getWidth());
      for (int y = 0; y < buffer->getHeight(); y++) {
         int fileIndex       = kIndex(xStart, y + yStart, fStart, nxGlobal, nyGlobal, nfGlobal);
         int bufferIndex     = kIndex(0, y, 0, nx, ny, nf);
         long currentFilePos = frameDataStart + (long)sizeof(T) * (long)fileIndex;
         fStream.setOutPos(currentFilePos, true /*fromBeginning flag*/);
         fStream.read(&bufferData[bufferIndex], lineWidth);
      }
   }
   else { // nfGlobal != bufferData.getFeatures();
      std::size_t dataWidth = sizeof(T) * (std::size_t)buffer->getFeatures();
      for (int y = 0; y < buffer->getHeight(); y++) {
         for (int x = 0; x < buffer->getWidth(); x++) {
            int fileIndex       = kIndex(xStart, y + yStart, fStart, nxGlobal, nyGlobal, nfGlobal);
            int bufferIndex     = kIndex(x, y, 0, nx, ny, nf);
            long currentFilePos = frameDataStart + (long)sizeof(T) * (long)fileIndex;
            fStream.setOutPos(currentFilePos, true /*fromBeginning flag*/);
            fStream.read(&bufferData[bufferIndex], dataWidth);
         }
      }
   }
   buffer->set(bufferData, nx, ny, nf);

   long frameDataEnd = frameDataStart + (long)sizeof(T) * (long)(nxGlobal * nyGlobal * nfGlobal);
   fStream.setOutPos(frameDataEnd, true /*fromBeginning flag*/);

   return timeStamp;
}

template <typename T>
HeaderDataType returnDataType() {
   // Specializations for byte types and taus_uint4 in BufferUtilsPvp.cpp
   if (std::numeric_limits<T>::is_integer) {
      return sizeof(T) == (std::size_t)1 ? BYTE : INT;
   }
   if (std::numeric_limits<T>::is_iec559) {
      return FLOAT;
   }
   return BufferUtils::UNRECOGNIZED_DATATYPE;
}

// Write a pvp header to fStream. After finishing, outStream will be pointing
// at the start of the first frame.
template <typename T>
ActivityHeader buildActivityHeader(int width, int height, int features, int numFrames) {
   HeaderDataType dataType = returnDataType<T>();
   FatalIf(
         dataType == UNRECOGNIZED_DATATYPE,
         "buildActivityHeader called with unrecognized data type.\n");

   ActivityHeader result;
   result.headerSize = sizeof(result);
   result.numParams  = result.headerSize / 4;
   result.fileType   = PVP_NONSPIKING_ACT_FILE_TYPE;
   result.nx         = width;
   result.ny         = height;
   result.nf         = features;
   result.numRecords = 1;
   result.recordSize = width * height * features;
   result.dataSize   = sizeof(T);
   result.dataType   = dataType;
   result.nxProcs    = 1;
   result.nyProcs    = 1;
   result.nxExtended = width;
   result.nyExtended = height;
   result.kx0        = 0;
   result.ky0        = 0;
   result.nBatch     = 1;
   result.nBands     = numFrames;
   result.timestamp  = 0;
   return result;
}

template <typename T>
ActivityHeader buildSparseActivityHeader(int width, int height, int features, int numFrames) {
   ActivityHeader header = buildActivityHeader<T>(width, height, features, numFrames);
   header.dataSize       = sizeof(struct SparseList<T>::Entry);
   header.fileType       = PVP_ACT_SPARSEVALUES_FILE_TYPE;
   header.recordSize     = 0;
   return header;
}

static void writeActivityHeader(FileStream &fStream, ActivityHeader const &header) {
   fStream.setOutPos(0, true);
   fStream.write(&header, sizeof(header));
}

// Reads a pvp header and returns it in vector format. Leaves inStream
// pointing at the start of the first frame.
static ActivityHeader readActivityHeader(FileStream &fStream) {
   fStream.setInPos(0, true);
   int headerSize = -1;
   fStream.read(&headerSize, sizeof(int));
   fStream.setInPos(0, true);
   ActivityHeader header;
   fStream.read(&header, headerSize);
   return header;
}

// Writes a buffer to a pvp file containing a header and a single frame.
// Use appendToPvp to write multiple frames to a pvp file.
template <typename T>
void writeToPvp(const char *fName, Buffer<T> *buffer, double timeStamp, bool verifyWrites) {
   FileStream fStream(fName, std::ios_base::out | std::ios_base::binary, verifyWrites);
   writeActivityHeader(
         fStream,
         buildActivityHeader<T>(buffer->getWidth(), buffer->getHeight(), buffer->getFeatures(), 1));
   writeFrame<T>(fStream, buffer, timeStamp);
}

template <typename T>
void appendToPvp(
      const char *fName,
      Buffer<T> *buffer,
      int frameWriteIndex,
      double timeStamp,
      bool verifyWrites) {
   FileStream fStream(
         fName, std::ios_base::out | std::ios_base::in | std::ios_base::binary, verifyWrites);

   // Modify the number of records in the header
   ActivityHeader header = readActivityHeader(fStream);
   FatalIf(
         frameWriteIndex > header.nBands,
         "Cannot write entry %d when only %d entries exist.\n",
         frameWriteIndex,
         header.nBands);
   header.nBands = frameWriteIndex + 1;
   writeActivityHeader(fStream, header);

   // fStream is now pointing at the first frame. Each frame is
   // the size of the timestamp (double) plus the size of the
   // frame's data (numElements * sizeof(T))
   long frameOffset = frameWriteIndex * (header.recordSize * header.dataSize + sizeof(double));

   fStream.setOutPos(frameOffset, false);
   writeFrame<T>(fStream, buffer, timeStamp);
}

template <typename T>
double readActivityFromPvp(
      char const *fName,
      Buffer<T> *buffer,
      int frameReadIndex,
      BufferUtils::SparseFileTable *sparseFileTable) {
   double timestamp;
   int fileType;
   {
      FileStream headerStream(fName, std::ios_base::in | std::ios_base::binary, false);
      struct BufferUtils::ActivityHeader header = BufferUtils::readActivityHeader(headerStream);
      fileType                                  = header.fileType;
   }
   switch (fileType) {
      case PVP_NONSPIKING_ACT_FILE_TYPE:
         timestamp = BufferUtils::readDenseFromPvp<T>(fName, buffer, frameReadIndex);
         break;
      case PVP_ACT_SPARSEVALUES_FILE_TYPE:
         timestamp = BufferUtils::readDenseFromSparsePvp<T>(
               fName, buffer, frameReadIndex, sparseFileTable);
         break;
      case PVP_ACT_FILE_TYPE:
         timestamp = BufferUtils::readDenseFromSparseBinaryPvp<T>(
               fName, buffer, frameReadIndex, sparseFileTable);
         break;
      default:
         Fatal().printf(
               "readActivityFromPvp: \"%s\" has file type %d, which is not an activity file "
               "type.\n",
               fName,
               fileType);
         break;
   }
   return timestamp;
}

template <typename T>
double readDenseFromPvp(const char *fName, Buffer<T> *buffer, int frameReadIndex) {
   FileStream fStream(fName, std::ios_base::in | std::ios_base::binary, false);
   ActivityHeader header = readActivityHeader(fStream);
   FatalIf(
         header.fileType != PVP_NONSPIKING_ACT_FILE_TYPE,
         "readDenseFromPvp() can only be used on non-sparse activity pvps "
         "(PVP_NONSPIKING_ACT_FILE_TYPE)\n");
   FatalIf(header.nBands <= 0, "\"%s\" header does not have a positive nbands field.\n", fName);
   buffer->resize(header.nx, header.ny, header.nf);
   int frameIndex   = frameReadIndex % header.nBands;
   long frameOffset = frameIndex * (header.recordSize * header.dataSize + sizeof(double));
   fStream.setInPos(frameOffset, false);
   return readFrame<T>(fStream, buffer);
}

// Writes a sparse frame (with values) to the current
// outstream location
template <typename T>
void writeSparseFrame(FileStream &fStream, SparseList<T> *list, double timeStamp) {
   size_t dataSize                              = sizeof(struct SparseList<T>::Entry);
   vector<struct SparseList<T>::Entry> contents = list->getContents();
   int numElements                              = contents.size();
   fStream.write(&timeStamp, sizeof(double));
   fStream.write(&numElements, sizeof(int));
   if (numElements > 0) {
      fStream.write(contents.data(), contents.size() * dataSize);
   }
}

// Reads a sparse frame (with values) from the current
// instream location
template <typename T>
double readSparseFrame(FileStream &fStream, SparseList<T> *list) {
   size_t dataSize  = sizeof(struct SparseList<T>::Entry);
   double timeStamp = -1;
   int numElements  = -1;
   fStream.read(&timeStamp, sizeof(double));
   fStream.read(&numElements, sizeof(int));
   FatalIf(timeStamp == -1, "Failed to read timeStamp.\n");
   FatalIf(numElements == -1, "Failed to read numElements.\n");
   vector<struct SparseList<T>::Entry> contents(numElements);
   if (numElements > 0) {
      fStream.read(contents.data(), contents.size() * dataSize);
   }
   list->set(contents);
   return timeStamp;
}

// Reads a sparse binary frame from the current
// instream location
template <typename T>
double readSparseBinaryFrame(FileStream &fStream, SparseList<T> *list, T oneValue) {
   double timeStamp = -1;
   int numElements  = -1;
   fStream.read(&timeStamp, sizeof(double));
   fStream.read(&numElements, sizeof(int));
   FatalIf(timeStamp == -1, "Failed to read timeStamp.\n");
   FatalIf(numElements == -1, "Failed to read numElements.\n");
   vector<struct SparseList<T>::Entry> contents(numElements);
   vector<int> indices(numElements);
   if (numElements > 0) {
      fStream.read(indices.data(), numElements * sizeof(int));
   }
   for (int i = 0; i < indices.size(); ++i) {
      contents.at(i).index = indices.at(i);
      contents.at(i).value = oneValue;
   }
   list->set(contents);
   return timeStamp;
}

// Builds a table of offsets and lengths for each pvp frame
// index up to (but not including) upToIndex. Works for both
// sparse activity and sparse binary files. Leaves the input
// stream pointing at the location where frame upToIndex would
// begin.
static SparseFileTable buildSparseFileTable(FileStream &fStream, int upToIndex) {
   ActivityHeader header = readActivityHeader(fStream);
   FatalIf(
         upToIndex > header.nBands,
         "buildSparseFileTable requested frame %d / %d.\n",
         upToIndex,
         header.nBands);

   SparseFileTable result;
   result.valuesIncluded = header.fileType != PVP_ACT_FILE_TYPE;
   int dataSize          = header.dataSize;
   result.frameLengths.resize(upToIndex + 1, 0);
   result.frameStartOffsets.resize(upToIndex + 1, 0);

   for (int f = 0; f < upToIndex + 1; ++f) {
      double timeStamp      = 0;
      long frameLength      = 0;
      long frameStartOffset = fStream.getInPos();
      fStream.read(&timeStamp, sizeof(double));
      fStream.read(&frameLength, sizeof(int));
      result.frameLengths.at(f)      = frameLength;
      result.frameStartOffsets.at(f) = frameStartOffset;
      if (f < upToIndex) {
         fStream.setInPos(frameLength * (long)dataSize, false);
      }
   }
   return result;
}

template <typename T>
void writeSparseToPvp(
      const char *fName,
      SparseList<T> *list,
      double timeStamp,
      int width,
      int height,
      int features,
      bool verifyWrites) {
   FileStream fStream(fName, std::ios_base::out | std::ios_base::binary, verifyWrites);
   writeActivityHeader(fStream, buildSparseActivityHeader<T>(width, height, features, 1));
   writeSparseFrame<T>(fStream, list, timeStamp);
}

template <typename T>
void appendSparseToPvp(
      const char *fName,
      SparseList<T> *list,
      double timeStamp,
      int frameWriteIndex,
      bool verifyWrites) {
   FileStream fStream(
         fName, std::ios_base::out | std::ios_base::in | std::ios_base::binary, verifyWrites);

   // Modify the number of records in the header
   ActivityHeader header = readActivityHeader(fStream);
   FatalIf(
         frameWriteIndex > header.nBands,
         "Cannot write entry %d when only %d entries exist.\n",
         frameWriteIndex,
         header.nBands);
   header.nBands = frameWriteIndex + 1;
   writeActivityHeader(fStream, header);

   SparseFileTable table = buildSparseFileTable(fStream, frameWriteIndex - 1);
   long frameOffset      = table.frameStartOffsets.at(frameWriteIndex - 1)
                      + table.frameLengths.at(frameWriteIndex - 1) * header.dataSize
                      + sizeof(double) + sizeof(int); // Time / numElements
   fStream.setOutPos(frameOffset, true);
   writeSparseFrame<T>(fStream, list, timeStamp);
}

template <typename T>
double readSparseFromPvp(
      const char *fName,
      SparseList<T> *list,
      int frameReadIndex,
      SparseFileTable *cachedTable) {
   FileStream fStream(fName, std::ios_base::in | std::ios_base::binary, false);

   ActivityHeader header = readActivityHeader(fStream);
   FatalIf(
         header.fileType != PVP_ACT_SPARSEVALUES_FILE_TYPE,
         "readSparseFromPvp() can only be used on sparse activity pvps "
         "(PVP_ACT_SPARSEVALUES_FILE_TYPE)\n");
   FatalIf(header.nBands <= 0, "\"%s\" header does not have a positive nbands field.\n", fName);
   FatalIf(
         header.dataSize != sizeof(struct SparseList<T>::Entry),
         "Error: Expected data size %d, found %d.\n",
         sizeof(struct SparseList<T>::Entry),
         header.dataSize);

   SparseFileTable table;
   if (cachedTable == nullptr) {
      table = buildSparseFileTable(fStream, frameReadIndex);
   }
   else {
      table = *cachedTable;
   }

   long frameOffset = table.frameStartOffsets.at(frameReadIndex);
   fStream.setInPos(frameOffset, true);
   return readSparseFrame<T>(fStream, list);
}

template <typename T>
double readDenseFromSparsePvp(
      char const *fName,
      Buffer<T> *buffer,
      int frameReadIndex,
      SparseFileTable *sparseFileTable) {
   SparseList<T> list;
   double timestamp = readSparseFromPvp(fName, &list, frameReadIndex, sparseFileTable);

   FileStream fStream(fName, std::ios_base::in | std::ios_base::binary, false);
   ActivityHeader header = readActivityHeader(fStream);
   buffer->resize(header.nx, header.ny, header.nf);
   list.toBuffer(*buffer, (T)0);

   return timestamp;
}

template <typename T>
double readSparseBinaryFromPvp(
      const char *fName,
      SparseList<T> *list,
      int frameReadIndex,
      T oneVal,
      SparseFileTable *cachedTable) {
   FileStream fStream(fName, std::ios_base::in | std::ios_base::binary, false);

   ActivityHeader header = readActivityHeader(fStream);
   FatalIf(
         header.fileType != PVP_ACT_FILE_TYPE,
         "readSparseBinaryFromPvp() can only be used on sparse binary pvps "
         "(PVP_ACT_FILE_TYPE)\n");
   FatalIf(header.nBands <= 0, "\"%s\" header does not have a positive nbands field.\n", fName);
   FatalIf(
         header.dataSize != sizeof(int),
         "Error: Expected data size %d, found %d.\n",
         sizeof(int),
         header.dataSize);

   SparseFileTable table;
   if (cachedTable == nullptr) {
      table = buildSparseFileTable(fStream, frameReadIndex);
   }
   else {
      table = *cachedTable;
   }

   long frameOffset = table.frameStartOffsets.at(frameReadIndex);
   fStream.setInPos(frameOffset, true);
   return readSparseBinaryFrame<T>(fStream, list, oneVal);
}

template <typename T>
double readDenseFromSparseBinaryPvp(
      char const *fName,
      Buffer<T> *buffer,
      int frameReadIndex,
      SparseFileTable *sparseFileTable) {
   SparseList<T> list;
   double timestamp = readSparseBinaryFromPvp(fName, &list, frameReadIndex, (T)1, sparseFileTable);

   FileStream fStream(fName, std::ios_base::in | std::ios_base::binary, false);
   ActivityHeader header = readActivityHeader(fStream);
   buffer->resize(header.nx, header.ny, header.nf);
   list.toBuffer(*buffer, (T)0);

   return timestamp;
}

template <typename T>
std::size_t weightPatchSize(int numWeightsInPatch) {
   HeaderDataType dataType = returnDataType<T>();
   FatalIf(
         dataType == UNRECOGNIZED_DATATYPE,
         "buildActivityHeader called with unrecognized data type.\n");

   std::size_t sz;
   switch (dataType) {
      case UNRECOGNIZED_DATATYPE: sz = (std::size_t)(-1); break;
      case BYTE: sz                  = sizeof(char); break;
      case INT: sz                   = sizeof(int); break;
      case FLOAT: sz                 = sizeof(float); break;
      default: pvAssert(0); break;
   }
   return (2 * sizeof(unsigned short) + sizeof(unsigned int) + numWeightsInPatch * sz);
}

} // end namespace BufferUtils
} // end namespace PV
