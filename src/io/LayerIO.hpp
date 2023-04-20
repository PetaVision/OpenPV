/*
 * LayerIO.hpp
 *
 *  Created on: April 15, 2021
 *      Author: peteschultz
 */

#ifndef LAYERIO_HPP_
#define LAYERIO_HPP_

#include "io/FileStream.hpp"
#include "structures/Buffer.hpp"
#include "utils/BufferUtilsPvp.hpp" // struct ActivityHeader

#include <memory>

namespace PV {

/**
 * A class to manage nonsparse activity PVP files.
 * Generally reading and writing a nonsparse activity PVP file should be done by means of this
 * class. The principal use case is by the LayerFile class, which creates and manages a
 * LayerIO class internally for its I/O operations.
 *
 * Opening a new file using LayerIO automatically writes the PVP header. The public function
 * members read() and write() transfer data as an entire frame.
 *
 * This class does not handle batching; see the LayerFile class regarding batching.
 *
 * The file position can only be set to the beginning of a frame, Setting the position to n
 * moves the file to the start of the (zero-indexed) nth frame.
 * For files opened in read/write mode, the read position and write position always move together.
 */
class LayerIO {
  public:
   LayerIO(std::shared_ptr<FileStream> fileStream, int width, int height, int numFeatures);

   virtual ~LayerIO() {}

   long calcFilePositionFromFrameNumber(int frameNumber) const;
   int calcFrameNumberFromFilePosition(long filePosition) const;

   void read(Buffer<float> &buffer);
   void read(Buffer<float> &buffer, double &timestamp);
   void read(Buffer<float> &buffer, double &timestamp, int frameNumber);

   void write(Buffer<float> const &buffer, double timestamp);
   void write(Buffer<float> const &buffer, double timestamp, int frameNumber);

   void open();
   void close();

   int getWidth() const { return mWidth; }
   int getHeight() const { return mHeight; }
   int getNumFeatures() const { return mNumFeatures; }

   int getFrameNumber() const { return mFrameNumber; }
   void setFrameNumber(int frame);
   int getNumFrames() const { return mNumFrames; }

   std::shared_ptr<FileStream> getFileStream() const { return mFileStream; }

  private:
   long calcValuesPerPVPFrame() const;

   // returns the number of bytes in the buffer data if it has the right dimensions;
   // fatal errors otherwise.
   long checkBufferDimensions(Buffer<float> const &buffer);
   void initializeNumFrames();
   void setHeaderNBands();
   void writeHeader();

  private:
   std::shared_ptr<FileStream> mFileStream;
   int mWidth;
   int mHeight;
   int mNumFeatures;

   int mFrameNumber = 0;
   int mNumFrames   = 0;

   long const mHeaderSize = static_cast<long>(sizeof(BufferUtils::ActivityHeader));
   long const mDataSize   = static_cast<long>(sizeof(float));
};

} // namespace PV

#endif // LAYERIO_HPP_
