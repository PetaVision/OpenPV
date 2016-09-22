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
      
      // Reads the specified frame from a pvp file. Returns the timeStamp.
      template <typename T>
      double readFrame(FileStream *fStream,
                                   Buffer<T> *buffer,
                                   int frameReadIndex) {
         vector<int> header = readHeader(fStream);
         vector<T> data(header.at(INDEX_RECORD_SIZE));
         pvErrorIf(frameReadIndex > header.at(INDEX_NBANDS),
               "readFrame requested frame %d, out of %d frames.\n",
               frameReadIndex, header.at(INDEX_NBANDS));
         std::streambuf::pos_type frameOffset =
               frameReadIndex *
                   ( header.at(INDEX_RECORD_SIZE)
                   * header.at(INDEX_DATA_SIZE)
                   + sizeof(double) );
      
         double timeStamp;
         fStream->inStream().seekg(frameOffset, std::ios_base::cur);
         fStream->inStream().read((char*)&timeStamp, sizeof(double));
      
         assert(fStream->inStream().gcount() == sizeof(double));
         
         size_t expected = data.size() * header.at(INDEX_DATA_SIZE);
         fStream->inStream().read((char*)data.data(), expected);
         assert(fStream->inStream().gcount() == expected);
         
         buffer->set(data,
                     header.at(INDEX_NX),
                     header.at(INDEX_NY),
                     header.at(INDEX_NF));
         return timeStamp;
      }
      
      // Write a pvp header to fStream. After finishing, outStream will be pointing
      // at the start of the first frame.
      template <typename T>
      vector<int> buildHeader(int width,
                                        int height,
                                        int features,
                                        int numFrames) {
         // TODO: This misses headersize
         vector<int> header(NUM_BIN_PARAMS);   
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
         FileStream *fStream =
            new FileStream(fName,
                           std::ios_base::out
                          |std::ios_base::binary,
                           false);
      
         pvErrorIf(fStream->outStream().bad(),
               "Failed to open ostream %s.\n", fName);
      
         writeHeader(fStream,
                     buildHeader<T>(buffer->getWidth(),
                                    buffer->getHeight(),
                                    buffer->getFeatures(),
                                    1));
         writeFrame<T>(fStream, buffer, timeStamp);
      }
      
      template <typename T>
      void appendToPvp(const char * fName,
                                 Buffer<T> *buffer,
                                 int frameWriteIndex,
                                 double timeStamp) {
         FileStream *fStream =
            new FileStream(fName,
                           std::ios_base::out
                          |std::ios_base::in
                          |std::ios_base::binary,
                           false);
      
         pvErrorIf(fStream->outStream().bad(),
               "Failed to open ostream %s.\n", fName);
         pvErrorIf(fStream->inStream().bad(),
               "Failed to open istream %s.\n", fName);
      
      
         // Modify the number of records in the header
         vector<int> header = readHeader(fStream);
         header.at(INDEX_NBANDS)++;
         writeHeader(fStream, header);
      
         // fStream is now pointing at the first frame. Each frame is
         // the size of the timestamp (double) plus the size of the
         // frame's data (numElements * sizeof(T))
         std::streambuf::pos_type frameOffset =
               frameWriteIndex *
                   ( header.at(INDEX_RECORD_SIZE)
                   * header.at(INDEX_DATA_SIZE)
                   + sizeof(double) );
      
         fStream->outStream().seekp(frameOffset, std::ios_base::cur);
         writeFrame<T>(fStream, buffer, timeStamp);
      }
      
      template <typename T>
      double readFromPvp(const char * fName,
                                 Buffer<T> *buffer,
                                 int frameReadIndex) {
         FileStream *fStream =
            new FileStream(fName,
                           std::ios_base::in
                          |std::ios_base::binary,
                           false);
      
         pvErrorIf(fStream->inStream().bad(),
               "Failed to open istream %s.\n", fName);
      
         double timeStamp = readFrame<T>(fStream, buffer, frameReadIndex);
         return timeStamp;
      }
   }
}
