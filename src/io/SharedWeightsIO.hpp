/*
 * SharedWeightsIO.hpp
 *
 *  Created on: April 15, 2021
 *      Author: peteschultz
 */

#ifndef LAYERIO_HPP_
#define LAYERIO_HPP_

#include "components/Weights.hpp"
#include "io/FileStream.hpp"
#include "utils/BufferUtilsPvp.hpp"

#include <memory>
#include <string>

namespace PV {

/**
 * A class to manage shared weight PVP files. 
 * Generally reading and writing a shared weight PVP file should be done by means of
 * this class. The principal use case is by the SharedWeightsFile class, which creates
 * and manages a SharedWeightsIO class internally for its I/O operations.
 *
 * Opening a new file using SharedWeightsIO creates an empty file. The public function members
 * read() and write() transfer data as an entire frame, which includes the header.
 *
 * The file position can only be set to the beginning of a frame, Setting the position to n
 * moves the file to the start of the (zero-indexed) nth frame.
 * For files opened in read/write mode, the read position and write position always move together.
 */
class SharedWeightsIO {
  public:
   SharedWeightsIO(std::shared_ptr<FileStream> fileStream);

   virtual ~SharedWeightsIO() {}

   long calcFilePositionFromFrameNumber(int frameNumber) const;
   int calcFrameNumberFromFilePosition(long filePosition) const;

   void read(Weights &weights);
   void read(Weights &weights, double &timestamp);
   void read(Weights &weights, double &timestamp, int frameNumber);

   void write(Weights &weights, bool compressed, double timestamp);
   void write(Weights &weights, bool compressed, double timestamp, int frameNumber);

   std::shared_ptr<FileStream> getFileStream() const { return mFileStream; }

   int getFrameNumber() const { return mFrameNumber; }
   void setFrameNumber(int frame);
   int getNumFrames() const { return mNumFrames; }

  private:
   long calcArborSize(BufferUtils::WeightHeader const &header) const;

   void checkHeaderValues(Weights const &weights, BufferUtils::WeightHeader const &header) const;

   void compressPatch(
         unsigned char *dataForFile,
         float const *sourceWeights,
         int count,
         float minValue,
         float maxValue);
   void decompressPatch(
         unsigned char const *readBuffer,
         float *destWeights,
         int count,
         float minValue,
         float maxValue);

   // returns the number of bytes in the buffer data if it has the right dimensions;
   // fatal errors otherwise.
   void initializeFrameStarts();

   void loadWeightsFromBuffer(
         Weights &weights,
         std::vector<unsigned char> const &readBuffer,
         int arbor,
         float minValue,
         float maxValue,
         bool compressed);
   void storeWeightsInBuffer(
         Weights const &weights,
         std::vector<unsigned char> &readBuffer,
         int arbor,
         float minValue,
         float maxValue,
         bool compressed);

   BufferUtils::WeightHeader writeHeader(
         Weights &weights, bool compressed, double timestamp, float minWeight, float maxWeight);

  private:
   std::shared_ptr<FileStream> mFileStream;

   int mFrameNumber = 0;
   int mNumFrames   = 0;

   long const mHeaderSize = static_cast<long>(sizeof(BufferUtils::WeightHeader));

   std::vector<long> mFrameStarts; // mFrameStarts[k] is the byte position of pvp frame n
   std::vector<int> mNumEntries;   // mNumEntries[k] is the number of nonzero values in pvp frame n
};

} // namespacePV

#endif // LAYERIO_HPP_
