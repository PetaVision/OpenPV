#include "io.hpp"

namespace PV {

// TODO: Check header[INDEX_FILE_TYPE] and error if it isn't supported


// Write a single frame to a pvp file, starting at fStream's location.
// A pvp file may contain multiple frames.
template <typename T>
void BufferIO::writeFrame(FileStream *fStream,
                              Buffer<T> *buffer,
                              double timestamp) {
   size_t dataSize = sizeof(T);
   pvErrorIf(!fStream->binary(),
         "writeBuffer requires a binary FileStream.\n");
   fStream->outStream().write(&timestamp, sizeof(double));
   fStream->outStream().write(buffer->asVector().data(),
                              buffer->getTotalElements() * dataSize);
   fStream->outStream().flush();
}

// Reads the specified frame from a pvp file. Returns the timeStamp.
template <typename T>
double BufferIO::readFrame(FileStream *fStream,
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
   buffer->resize(header.at(INDEX_NX),
                  header.at(INDEX_NY),
                  header.at(INDEX_NF));
   double timeStamp;
   fStream->outStream().seekp(frameOffset, std::ios_base::cur);
   fStream->outStream().read(&timeStamp, sizeof(double));
   fStream->outStream().read(buffer->asVector().data(),
                             buffer->getTotalElements()
                           * header.at(INDEX_DATA_SIZE));
   return timeStamp;
}

// Write a pvp header to fStream. After finishing, outStream will be pointing
// at the start of the first frame.
template <typename T>
vector<int> BufferIO::buildHeader(int width,
                                  int height,
                                  int features,
                                  int numFrames) {
   vector<uint32_t> header(NUM_BIN_PARAMS);   
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

void BufferIO::writeHeader(FileStream *fStream, vector<int> header) {
   pvErrorIf(!fStream->binary(),
         "writeBuffer requires a binary FileStream.\n");
   fStream->outStream().seekp(std::ios_base::beg);
   fStream->outStream().write(header.data(),
                              header.size() * sizeof(uint32_t));
   fStream->outStream().flush();
}
 
// Reads a pvp header and returns it in vector format. Leaves inStream
// pointing at the start of the first frame.
vector<int> BufferIO::readHeader(FileStream *fStream) {
   vector<int> header(NUM_BIN_PARAMS);
   fStream->inStream().seekg(0, fStream->inStream().beg);
   fStream->inStream().read(header.data(), header.size() * sizeof(uint32_t));
   return header;
}

// TODO: Allow verify writes for these

// Writes a buffer to a pvp file containing a header and a single frame.
// Use appendToPvp to write multiple frames to a pvp file.
template <typename T>
void BufferIO::writeToPvp(string fName,
                          Buffer<T> *buffer,
                          double timeStamp) {
   FileStream *fStream =
      new FileStream(fName.c_str(),
                     std::ios_base::out
                    |std::ios_base::binary,
                     false);
   // TODO: Make sure the file was opened successfully
   writeHeader(fStream,
               buildHeader<T>(buffer->getWidth(),
                              buffer->getHeight(),
                              buffer->getFeatures(),
                              1));
   writeFrame<T>(fStream, buffer, timeStamp);
   fStream->closeFile();
}

template <typename T>
void BufferIO::appendToPvp(string fName,
                           Buffer<T> *buffer,
                           int frameWriteIndex,
                           double timeStamp) {
   FileStream *fStream =
      new FileStream(fName.c_str(),
                     std::ios_base::out
                    |std::ios_base::in
                    |std::ios_base::binary,
                     false);
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
   fStream->closeFile();
}

template <typename T>
double BufferIO::readFromPvp(string fName,
                           Buffer<T> *buffer,
                           int frameReadIndex) {
   FileStream *fStream =
      new FileStream(fName.c_str(),
                     std::ios_base::in
                    |std::ios_base::binary,
                     false);
   double timeStamp = readFrame<T>(fStream, buffer, frameReadIndex);
   fStream->closeFile();
   return timeStamp;
}

}
