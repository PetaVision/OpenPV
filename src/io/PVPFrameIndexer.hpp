#ifndef PVPFRAMEINDEXER_HPP_
#define PVPFRAMEINDEXER_HPP_

#include "io/FileStream.hpp"

#include <memory>

namespace PV {

class PVPFrameIndexer {
  public:
   PVPFrameIndexer(
         std::shared_ptr<FileStream> fileStream,
         long frameSize,
         long externalHeaderSize);

   virtual ~PVPFrameIndexer();

   long getExternalHeaderSize() const { return mExternalHeaderSize; }

   /**
    * Calculate the file position from the indicated frameNumber, which is the frameNumber times
    * the size of one frame in bytes (see the private function member calcFrameSizeBytes()).
    * @details This is useful when writing checkpoints, because we checkpoint the file position
    * as a byte offset, not the frame number.
    */
   long calcFilePositionFromFrameNumber(long frameNumber) const;
         
   /**
    * Calculate the frame number from the indicated filePosition, which is the filePosition
    * divided by the size of one frame in bytes, discarding any remainder.
    * For the frame size, see the private function member calcFrameSizeBytes().
    * @details This is useful when reading checkpoints, because we checkpoint the file position
    * as a byte offset, not the frame number.
    */
   long calcFrameNumberFromFilePosition(long filePosition) const;

   void moveFilePosToFrameStart();

   std::shared_ptr<FileStream> getFileStream() const { return mFileStream; }

   int getFrameNumber() const { return mFrameNumber; }
   void setFrameNumber(int frameNumber);

   long getFrameSize() const { return mFrameSize; }

   int getNumFrames() const { return mNumFrames; }

  private:
   void initializeNumFrames();
   int convertToLogicalFrameNumber(int frameNumber);

  private:
   long mExternalHeaderSize;
   std::shared_ptr<FileStream> mFileStream = nullptr;
   int mFrameNumber   = 0;
   long mFrameSize    = 0L;
   int mNumFrames     = 0;
   bool mReadOnlyFlag = false;
};

} // namespace PV

#endif // PVPFRAMEINDEXER_HPP_
