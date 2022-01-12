/*
 * SharedWeightsIO.hpp
 *
 *  Created on: June 29, 2021
 *      Author: peteschultz
 */

#ifndef SHAREDWEIGHTSIO_HPP_
#define SHAREDWEIGHTSIO_HPP_

#include "io/FileStream.hpp"
#include "utils/BufferUtilsPvp.hpp" // struct WeightHeader
#include "structures/Buffer.hpp"
#include "structures/Patch.hpp"
#include "structures/WeightData.hpp"

#include <memory>

namespace PV {

/**
 * A class to manage shared weight PVP files. 
 * Generally reading and writing a shared weight PVP file should be done by means of this class.
 * The principal use case is by the SharedWeightsFile class, which creates and manages a
 * SharedWeightsIO class internally for its I/O operations.
 *
 * Opening a new file using SharedWeightsIO creates an empty file. The public function members
 * read() loads all the weights from the frame. The write() function writes the header as well
 * as all the weights to the frame.
 *
 * The file position can only be set to the beginning of a frame, Setting the position to index n
 * moves the file to the start of the (zero-indexed) nth frame.
 * For files opened in read/write mode, the read position and write position always move together.
 */
class SharedWeightsIO {
  public:
   SharedWeightsIO(
      std::shared_ptr<FileStream> fileStream,
      int patchSizeX,
      int patchSizeY,
      int patchSizeF,
      int numPatchesX,
      int numPatchesY,
      int numPatchesF,
      int numArbors,
      bool compressedFlag);

   virtual ~SharedWeightsIO() {}

   long calcFilePositionFromFrameNumber(int frameNumber) const;
   int calcFrameNumberFromFilePosition(long filePosition) const;

   void read(WeightData &weightData);
   void read(WeightData &weightData, double &timestamp);
   void read(WeightData &weightData, double &timestamp, int frameNumber);

   void write(WeightData const &weightData, double timestamp);
   void write(WeightData const &weightData, double timestamp, int frameNumber);

   std::shared_ptr<FileStream> getFileStream() const { return mFileStream; }

   int getPatchSizeX() const { return mPatchSizeX; }
   int getPatchSizeY() const { return mPatchSizeY; }
   int getPatchSizeF() const { return mPatchSizeF; }
   long getPatchSizeOverall() const { return mPatchSizeX * mPatchSizeY * mPatchSizeF; }

   int getNumPatchesX() const { return mNumPatchesX; }
   int getNumPatchesY() const { return mNumPatchesY; }
   int getNumPatchesF() const { return mNumPatchesF; }
   long getNumPatchesOverall() const { return mNumPatchesX * mNumPatchesY * mNumPatchesF; }
   int getNumArbors() const { return mNumArbors; }
   bool getCompressedFlag() const { return mCompressedFlag; }

   long int getFrameSize() const { return mFrameSize; }
   int getFrameNumber() const { return mFrameNumber; }
   void setFrameNumber(int frame);
   int getNumFrames() const { return mNumFrames; }

  private:
   void calcExtremeWeights(WeightData const &weightData, float &minWeight, float &maxWeight) const;

   void checkDimensions(WeightData const &weightData);
   void checkHeader(BufferUtils::WeightHeader const &header) const;

   void initializeFrameSize();
   void initializeNumFrames();
   double readInternal(WeightData &weightData);
   // void setHeaderNBands(); // We might do this for weights as we do for layers; for now we don't
   void writeHeader();

  private:
   std::shared_ptr<FileStream> mFileStream;
   int mPatchSizeX;
   int mPatchSizeY;
   int mPatchSizeF;
   int mNumPatchesX;
   int mNumPatchesY;
   int mNumPatchesF;
   int mNumArbors;
   bool mCompressedFlag;

   long mDataSize = static_cast<long>(sizeof(float));

   long mFrameSize  = 0L; // Number of bytes in a frame, including the header
   int mFrameNumber     = 0;
   int mNumFrames       = 0;

   long const mHeaderSize = static_cast<long>(sizeof(BufferUtils::WeightHeader));
   long const mPatchHeaderSize = static_cast<long>(sizeof(Patch));
};

} // namespacePV

#endif // SHAREDWEIGHTSIO_HPP_
