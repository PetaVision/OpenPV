#include "io/io.hpp"

namespace PV {

   // TODO: Check header[INDEX_FILE_TYPE] and error if it isn't supported
   
   namespace BufferUtils {

      // Write a single frame to a pvp file, starting at fStream's location.
      // A pvp file may contain multiple frames.
      template <typename T>
      void writeFrame(FileStream &fStream,
                      Buffer<T> *buffer,
                      double timestamp) {
         size_t dataSize = sizeof(T);
         fStream.write(&timestamp, sizeof(double));
         fStream.write(buffer->asVector().data(),
                       buffer->getTotalElements() * dataSize);
      }
      
      // Reads the next frame from a pvp file. Returns the timeStamp.
      // Assumes that buffer is already the correct dimensions for
      // the expected data.
      template <typename T>
      double readFrame(FileStream &fStream,
                       Buffer<T> *buffer) {
         double timeStamp;
         fStream.read(&timeStamp, sizeof(double));
         
         vector<T> data(buffer->getTotalElements());
         size_t expected = data.size() * sizeof(T);
         fStream.read(data.data(), expected);
         
         buffer->set(data,
                     buffer->getWidth(),
                     buffer->getHeight(),
                     buffer->getFeatures());
         return timeStamp;
      }
      
      // Write a pvp header to fStream. After finishing, outStream will be pointing
      // at the start of the first frame.
      template <typename T>
      vector<int> buildHeader(int width,
                              int height,
                              int features,
                              int numFrames,
                              bool isSparse) {
         vector<int> header(NUM_BIN_PARAMS);
         header.at(INDEX_HEADER_SIZE) = header.size() * sizeof(int);
         header.at(INDEX_FILE_TYPE)   = PVP_NONSPIKING_ACT_FILE_TYPE;
         header.at(INDEX_NX)          = width;
         header.at(INDEX_NY)          = height;
         header.at(INDEX_NF)          = features;
         header.at(INDEX_NUM_RECORDS) = 1;
         header.at(INDEX_RECORD_SIZE) = width * height * features;
         header.at(INDEX_DATA_SIZE)   = isSparse
                                        ? sizeof(struct SparseList<T>::Entry)
                                        : sizeof(T);
         header.at(INDEX_DATA_TYPE)   = PV_FLOAT_TYPE; // TODO: How to template this?
         header.at(INDEX_NX_PROCS)    = 1;
         header.at(INDEX_NY_PROCS)    = 1;
         header.at(INDEX_NX_GLOBAL)   = width;
         header.at(INDEX_NY_GLOBAL)   = height;
         header.at(INDEX_KX0)         = 0;
         header.at(INDEX_KY0)         = 0;
         header.at(INDEX_NBATCH)      = 1;
         header.at(INDEX_NBANDS)      = numFrames;
         header.at(INDEX_TIME)        = 0;
         header.at(INDEX_TIME+1)      = 0;
         return header;
      }
      
      void writeHeader(FileStream &fStream, vector<int> header) {
         fStream.setOutPos(0, true);
         fStream.write(header.data(), header.size() * sizeof(int));
      }
       
      // Reads a pvp header and returns it in vector format. Leaves inStream
      // pointing at the start of the first frame.
      vector<int> readHeader(FileStream &fStream) {
         vector<int> header(NUM_BIN_PARAMS);
         fStream.setInPos(0, true);
         fStream.read(header.data(), header.size() * sizeof(int));
         return header;
      }
      
      // Writes a buffer to a pvp file containing a header and a single frame.
      // Use appendToPvp to write multiple frames to a pvp file.
      template <typename T>
      void writeToPvp(const char * fName,
                      Buffer<T> *buffer,
                      double timeStamp) {
         FileStream fStream(fName,
                            std::ios_base::out
                          | std::ios_base::binary,
                            false);
     
         writeHeader(fStream, buildHeader<T>(buffer->getWidth(),
                                             buffer->getHeight(),
                                             buffer->getFeatures(),
                                             1,
                                             false));
         writeFrame<T>(fStream, buffer, timeStamp);
      }
      
      template <typename T>
      void appendToPvp(const char * fName,
                       Buffer<T> *buffer,
                       int frameWriteIndex,
                       double timeStamp) {
         FileStream fStream(fName,
                            std::ios_base::out
                          | std::ios_base::in
                          | std::ios_base::binary,
                            false);
      
         // TODO: Error if we're writing more than 1 index past the end

         // Modify the number of records in the header
         vector<int> header = readHeader(fStream);
         header.at(INDEX_NBANDS) = frameWriteIndex + 1;
         writeHeader(fStream, header);
      
         // fStream is now pointing at the first frame. Each frame is
         // the size of the timestamp (double) plus the size of the
         // frame's data (numElements * sizeof(T))
         long frameOffset = frameWriteIndex *
                              ( header.at(INDEX_RECORD_SIZE)
                              * header.at(INDEX_DATA_SIZE)
                              + sizeof(double) );
      
         fStream.setOutPos(frameOffset, false);
         writeFrame<T>(fStream, buffer, timeStamp);
      }
      
      template <typename T>
      double readFromPvp(const char * fName,
                         Buffer<T> *buffer,
                         int frameReadIndex) {
         FileStream fStream(fName,
                            std::ios_base::in
                          | std::ios_base::binary,
                            false);
         vector<int> header = readHeader(fStream);
         buffer->resize(header.at(INDEX_NX),
                        header.at(INDEX_NY),
                        header.at(INDEX_NF));
         long frameOffset = frameReadIndex *
                              ( header.at(INDEX_RECORD_SIZE)
                              * header.at(INDEX_DATA_SIZE)
                              + sizeof(double) );
         fStream.setInPos(frameOffset, false);
         return readFrame<T>(fStream, buffer);
      }

      // Writes a sparse frame (with values) to the current
      // outstream location
      template <typename T>
      void writeSparseFrame(FileStream &fStream,
                            SparseList<T> *list,
                            double timeStamp) {
         size_t dataSize = sizeof(struct SparseList<T>::Entry);
         vector<struct SparseList<T>::Entry> contents = list->getContents();
         int numElements = contents.size();
         fStream.write(&timeStamp,   sizeof(double));
         fStream.write(&numElements, sizeof(int));
         if (numElements > 0) {
            fStream.write(contents.data(), contents.size() * dataSize);
         }
      }

      // Reads a sparse frame (with values) from the current
      // instream location
      template <typename T>
      double readSparseFrame(FileStream &fStream,
                             SparseList<T> *list) {
         size_t dataSize    = sizeof(struct SparseList<T>::Entry);
         double timeStamp   = -1;
         int    numElements = -1;
         fStream.read(&timeStamp,   sizeof(double));
         fStream.read(&numElements, sizeof(int));
         pvErrorIf(timeStamp == -1,
               "Failed to read timeStamp.\n");
         pvErrorIf(numElements == -1,
               "Failed to read numElements.\n");
         vector<struct SparseList<T>::Entry> contents(numElements);
         if (numElements > 0) {
            fStream.read(contents.data(), contents.size() * dataSize);
         }
         list->set(contents);
         return timeStamp;
      }

      // Builds a table of offsets and lengths for each pvp frame
      // index up to (but not including) upToIndex. Works for both
      // sparse activity and sparse binary files. Leaves the input
      // stream pointing at the location where frame upToIndex would
      // begin.
      static SparseFileTable buildSparseFileTable(FileStream &fStream,
                                                 int upToIndex) {
         vector<int> header = readHeader(fStream);
         pvErrorIf(upToIndex > header.at(INDEX_NBANDS),
               "buildSparseFileTable requested frame %d / %d.\n",
               upToIndex, header.at(INDEX_NBANDS));

         SparseFileTable result;
         result.valuesIncluded = header.at(INDEX_FILE_TYPE) != PVP_ACT_FILE_TYPE;
         int dataSize = header.at(INDEX_DATA_SIZE);
         result.frameLengths.resize(upToIndex+1, 0);
         result.frameStartOffsets.resize(upToIndex+1, 0);

         for (int f = 0; f < upToIndex+1; ++f) {
            double timeStamp        = 0;
            int    frameLength      = 0;
            long   frameStartOffset = fStream.getInPos();
            fStream.read(&timeStamp,   sizeof(double));
            fStream.read(&frameLength, sizeof(int));
            result.frameLengths.at(f)      = frameLength;
            result.frameStartOffsets.at(f) = frameStartOffset;
            if (f < upToIndex) {
               fStream.setInPos(frameLength * dataSize, false);
            }
         }
         return result;
      }

      template <typename T>
      static void writeSparseToPvp(const char *fName,
                                   SparseList<T> *list,
                                   double timeStamp,
                                   int width,
                                   int height,
                                   int features) {
         FileStream fStream(fName,
                            std::ios_base::out
                          | std::ios_base::binary,
                            false);
         writeHeader(fStream, buildHeader<T>(width,
                                             height,
                                             features,
                                             1,
                                             true));
         writeSparseFrame<T>(fStream, list, timeStamp); 
      }

      template <typename T>
      static void appendSparseToPvp(const char *fName,
                                    SparseList<T> *list,
                                    double timeStamp,
                                    int frameWriteIndex) {
         FileStream fStream(fName,
                            std::ios_base::out
                          | std::ios_base::in
                          | std::ios_base::binary,
                            false);
         // Modify the number of records in the header
         vector<int> header = readHeader(fStream);
         header.at(INDEX_NBANDS) = frameWriteIndex + 1;
         writeHeader(fStream, header);
 
         SparseFileTable table = buildSparseFileTable(fStream,
                                                      frameWriteIndex - 1);
         long frameOffset = table.frameStartOffsets.at(frameWriteIndex - 1)
                          + table.frameLengths.at(frameWriteIndex - 1)
                          * header.at(INDEX_DATA_SIZE)
                          + sizeof(double) + sizeof(int); // Time / numElements
         fStream.setOutPos(frameOffset, true);
         writeSparseFrame<T>(fStream, list, timeStamp); 
      }

      template <typename T>
      static double readSparseFromPvp(const char *fName,
                                    SparseList<T> *list,
                                    int frameReadIndex,
                                    SparseFileTable *cachedTable = nullptr) {
         FileStream fStream(fName,
                            std::ios_base::in
                          | std::ios_base::binary,
                            false);

         vector<int> header = readHeader(fStream);
         pvErrorIf(header.at(INDEX_DATA_SIZE)
                != sizeof(struct SparseList<T>::Entry),
                "Error: Expected data size %d, found %d.\n",
                sizeof(struct SparseList<T>::Entry),
                header.at(INDEX_DATA_SIZE));

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
   }
}