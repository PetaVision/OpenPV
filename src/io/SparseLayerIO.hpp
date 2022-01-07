/*
 * SparseLayerIO.hpp
 *
 *  Created on: May 6, 2021
 *      Author: peteschultz
 */

#ifndef SPARSELAYERIO_HPP_
#define SPARSELAYERIO_HPP_

#include "io/FileStream.hpp"
#include "structures/SparseList.hpp" // struct ActivityHeader
#include "utils/BufferUtilsPvp.hpp" // struct ActivityHeader

#include <memory>
#include <string>

namespace PV {

/**
 * A class to manage sparse activity PVP files. 
 * Generally reading and writing a sparse activity PVP file should be done by means of this
 * class. The principal use case is by the SparseLayerFile class, which creates and manages a
 * SparseLayerIO class internally for its I/O operations.
 *
 * Opening a new file using SparseLayerIO automatically writes the PVP header. The public
 * function members read() and write() transfer data as an entire frame.
 *
 * This class does not handle batching; see the LayerFile class regarding batching.
 *
 * The file position can only be set to the beginning of a frame, Setting the position to n
 * moves the file to the start of the (zero-indexed) nth frame.
 * For files opened in read/write mode, the read position and write position always move together.
 */
class SparseLayerIO {
  public:
   SparseLayerIO(
      std::shared_ptr<FileStream> fileStream,
      int width,
      int height,
      int numFeatures);

   virtual ~SparseLayerIO() {}

   long calcFilePositionFromFrameNumber(int frameNumber) const;
   int calcFrameNumberFromFilePosition(long filePosition) const;

   void read(SparseList<float> &sparseList);
   void read(SparseList<float> &sparseList, double &timestamp);
   void read(SparseList<float> &sparseList, double &timestamp, int frameNumber);

   void write(SparseList<float> const &sparseList, double timestamp);
   void write(SparseList<float> const &sparseList, double timestamp, int frameNumber);

   std::shared_ptr<FileStream> getFileStream() const { return mFileStream; }

   int getWidth() const { return mWidth; }
   int getHeight() const { return mHeight; }
   int getNumFeatures() const { return mNumFeatures; }

   int getFrameNumber() const { return mFrameNumber; }
   void setFrameNumber(int frame);
   int getNumFrames() const { return mNumFrames; }

  private:
   void initializeFrameStarts();
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
   long const mSparseValueEntrySize = static_cast<long>(sizeof(SparseList<float>::Entry));

   std::vector<long> mFrameStarts; // mFrameStarts[k] is the byte position of pvp frame n
   std::vector<int> mNumEntries;   // mNumEntries[k] is the number of nonzero values in pvp frame n

   // FrameStarts always has length NumFrames + 1, with the last entry being the end-of-file position.
   // NumEntries always has length NumFrames

}; // class SparseLayerIO

} // namespacePV

#endif // SPARSELAYERIO_HPP_
