#include "io/io.hpp"

namespace PV {

   // TODO: Check header[INDEX_FILE_TYPE] and error if it isn't supported
   
   namespace BufferUtils {

      // Write a single frame to a pvp file, starting at fStream's location.
      // A pvp file may contain multiple frames.
      template <typename T>
      void writeFrame(FileStream *fStream,
                      Buffer<T> *buffer,
                      double timestamp) {
         size_t dataSize = sizeof(T);
         pvErrorIf(!fStream->binary(),
               "writeBuffer requires a binary FileStream.\n");
         fStream->outStream().write((char*)&timestamp, sizeof(double));
         fStream->outStream().write((char*)buffer->asVector().data(),
                                    buffer->getTotalElements() * dataSize);
         fStream->outStream().flush();
      }
      
      // Reads the next frame from a pvp file. Returns the timeStamp.
      // Assumes that buffer is already the correct dimensions for
      // the expected data.
      template <typename T>
      double readFrame(FileStream *fStream,
                       Buffer<T> *buffer) {
         double timeStamp;
         fStream->inStream().read((char*)&timeStamp, sizeof(double));
         assert(fStream->inStream().gcount() == sizeof(double));
         
         vector<T> data(buffer->getTotalElements());
         size_t expected = data.size() * sizeof(T);
         fStream->inStream().read((char*)data.data(), expected);
         assert(fStream->inStream().gcount() == expected);
         
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
                              int numFrames) {
         vector<int> header(NUM_BIN_PARAMS);
         header.at(INDEX_HEADER_SIZE) = header.size() * sizeof(int);
         header.at(INDEX_FILE_TYPE)   = PVP_NONSPIKING_ACT_FILE_TYPE;
         header.at(INDEX_NX)          = width;
         header.at(INDEX_NY)          = height;
         header.at(INDEX_NF)          = features;
         header.at(INDEX_NUM_RECORDS) = 1;
         header.at(INDEX_RECORD_SIZE) = width * height * features;
         header.at(INDEX_DATA_SIZE)   = sizeof(T);
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
      
      void writeHeader(FileStream *fStream, vector<int> header) {
         pvErrorIf(!fStream->binary(),
               "writeBuffer requires a binary FileStream.\n");
         fStream->outStream().seekp(std::ios_base::beg);
         fStream->outStream().write((char*)header.data(),
                                    header.size() * sizeof(int));
         fStream->outStream().flush();
      }
       
      // Reads a pvp header and returns it in vector format. Leaves inStream
      // pointing at the start of the first frame.
      vector<int> readHeader(FileStream *fStream) {
         vector<int> header(NUM_BIN_PARAMS);
         fStream->inStream().seekg(0);
         fStream->inStream().read((char*)header.data(), header.size() * sizeof(int));
         return header;
      }
      
      // TODO: Allow verify writes for these
      
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
      
         pvErrorIf(fStream.outStream().bad(),
               "Failed to open ostream %s.\n", fName);
      
         writeHeader(&fStream,
                     buildHeader<T>(buffer->getWidth(),
                                    buffer->getHeight(),
                                    buffer->getFeatures(),
                                    1));
         writeFrame<T>(&fStream, buffer, timeStamp);
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
      
         pvErrorIf(fStream.outStream().bad(),
               "Failed to open ostream %s.\n", fName);
         pvErrorIf(fStream.inStream().bad(),
               "Failed to open istream %s.\n", fName);
      
         // TODO: Error if we're writing more than 1 index past the end

         // Modify the number of records in the header
         vector<int> header = readHeader(&fStream);
         header.at(INDEX_NBANDS) = frameWriteIndex + 1;
         writeHeader(&fStream, header);
      
         // fStream is now pointing at the first frame. Each frame is
         // the size of the timestamp (double) plus the size of the
         // frame's data (numElements * sizeof(T))
         std::streambuf::pos_type frameOffset =
               frameWriteIndex *
                   ( header.at(INDEX_RECORD_SIZE)
                   * header.at(INDEX_DATA_SIZE)
                   + sizeof(double) );
      
         fStream.outStream().seekp(frameOffset, std::ios_base::cur);
         writeFrame<T>(&fStream, buffer, timeStamp);
      }
      
      template <typename T>
      double readFromPvp(const char * fName,
                         Buffer<T> *buffer,
                         int frameReadIndex) {
         FileStream fStream(fName,
                            std::ios_base::in
                          | std::ios_base::binary,
                            false);
      
         pvErrorIf(fStream.inStream().bad(),
               "Failed to open istream %s.\n", fName);
     
         vector<int> header = readHeader(&fStream);
         buffer->resize(header.at(INDEX_NX),
                        header.at(INDEX_NY),
                        header.at(INDEX_NF));
         std::streambuf::pos_type frameOffset = frameReadIndex *
                                              ( header.at(INDEX_RECORD_SIZE)
                                              * header.at(INDEX_DATA_SIZE)
                                              + sizeof(double) );
         fStream.inStream().seekg(frameOffset, std::ios_base::cur);
         double timeStamp = readFrame<T>(&fStream, buffer);
         return timeStamp;
      }

      // Writes a sparse frame (with values) to the current
      // outstream location
      template <typename T>
      void writeSparseFrame(FileStream *fStream,
                            SparseList<T> *list,
                            double timestamp) {
         size_t dataSize = sizeof(struct SparseList<T>::Entry);
         vector<struct SparseList<T>::Entry> contents = list->getContents();
         pvErrorIf(!fStream->binary(),
               "writeSparseFrame requires a binary FileStream.\n");
         fStream->outStream().write((char*)&timestamp, sizeof(double));
         fStream->outStream().write((char*)contents.data(),
                                    contents.size() * dataSize);
         fStream->outStream().flush();
      }

      // Reads a sparse frame (with values) from the current
      // instream location
      template <typename T>
      double readSparseFrame(FileStream *fStream,
                             SparseList<T> *list) {
         size_t dataSize    = sizeof(struct SparseList<T>::Entry);
         double timeStamp   = 0;
         int    numElements = 0;
         fStream->inStream().read((char*)&timeStamp, sizeof(double));
         fStream->inStream().read((char*)&numElements, sizeof(int));
         vector<struct SparseList<T>::Entry> contents(numElements);
         if (numElements > 0) {
            fStream->inStream().read((char*)contents.data(),
                                     contents.size() * dataSize);
         }
         list->set(contents);
         return timeStamp;
      }

      // Builds a table of offsets and lengths for each pvp frame
      // index up to (but not including) upToIndex. Works for both
      // sparse activity and sparse binary files. Leaves the input
      // stream pointing at the location where frame upToIndex would
      // begin.
      static SparseFileTable buildSparseFileTable(FileStream *fStream,
                                                 int upToIndex) {
         vector<int> header = readHeader(fStream);
         pvErrorIf(upToIndex > header.at(INDEX_NBANDS),
               "buildSparseFileTable requested frame %d / %d.\n",
               upToIndex, header.at(INDEX_NBANDS));

         SparseFileTable result;
         result.valuesIncluded = header.at(INDEX_FILE_TYPE) != PVP_ACT_FILE_TYPE;
         int dataSize = sizeof(int); // Indices are stored as ints
         if (result.valuesIncluded) {
            dataSize += header.at(INDEX_DATA_SIZE);
         }

         result.frameLengths.resize(upToIndex);
         result.frameStartOffsets.resize(upToIndex);

         for (int f = 0; f < upToIndex; ++f) {
            double timeStamp        = 0;
            int    frameLength      = 0;
            long   frameStartOffset = fStream->inStream().tellg();
            fStream->inStream().read((char*)&timeStamp, sizeof(double));
            fStream->inStream().read((char*)&frameLength, sizeof(int));
            result.frameLengths.at(f)      = frameLength;
            result.frameStartOffsets.at(f) = frameStartOffset;
            fStream->inStream().seekg(frameLength * dataSize,
                                      std::ios_base::cur);
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

         pvErrorIf(fStream.outStream().bad(),
               "Failed to open ostream %s.\n", fName);
      
         writeHeader(&fStream,
                     buildHeader<T>(width,
                                    height,
                                    features,
                                    1));
         writeSparseFrame<T>(&fStream, list, timeStamp); 
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
         pvErrorIf(fStream.outStream().bad(),
               "Failed to open ostream %s.\n", fName);
         pvErrorIf(fStream.inStream().bad(),
               "Failed to open istream %s.\n", fName);

         // Modify the number of records in the header
         vector<int> header = readHeader(&fStream);
         header.at(INDEX_NBANDS) = frameWriteIndex + 1;
         writeHeader(&fStream, header);
 
         SparseFileTable table = buildSparseFileTable(&fStream,
                                                      frameWriteIndex);
         std::streambuf::pos_type frameOffset =
            table.frameStartOffsets.at(frameWriteIndex - 1)
          + table.frameLengths.at(frameWriteIndex - 1);

         fStream.outStream().seekp(frameOffset, std::ios_base::cur);
         writeSparseFrame<T>(&fStream, list, timeStamp); 
      }
   }
}
