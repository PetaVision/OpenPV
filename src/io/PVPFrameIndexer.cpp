#include "PVPFrameIndexer.hpp"

#include <ios>
#include <stdexcept>
#include <string>

namespace PV {

PVPFrameIndexer::PVPFrameIndexer(
      std::shared_ptr<FileStream> fileStream,
      long frameSize,
      long externalHeaderSize) {
   if (frameSize <= 0L) {
      throw std::invalid_argument("PVPFrameIndexer called with nonpositive FrameSize argument\n");
   }
   mFileStream         = fileStream;
   mFrameSize          = frameSize;
   mExternalHeaderSize = externalHeaderSize;
   if (fileStream == nullptr) { return; }
   mReadOnlyFlag       = !fileStream->writeable();
   initializeNumFrames();

   // If writeable, initialize position at end of file.
   // If read-only, initialize position at beginning.
   // Users can call setFrameNumber() if something else is desired.
   if (mReadOnlyFlag) {
      setFrameNumber(0);
   }
   else {
      setFrameNumber(getNumFrames());
   }
}

PVPFrameIndexer::~PVPFrameIndexer() {}

void PVPFrameIndexer::initializeNumFrames() {
   if (mFileStream == nullptr) { return; }
   long currentPos = mFileStream->getInPos();
   mFileStream->setInPos(0L, std::ios_base::end);
   long fileSize = mFileStream->getInPos();
   mFileStream->setInPos(currentPos, std::ios_base::beg);
   if (fileSize > 0L) {
      mNumFrames = static_cast<int>((fileSize - mExternalHeaderSize) / mFrameSize);
      if (mNumFrames * mFrameSize != fileSize) {
         std::string errMsg(
               "PVPFrameIndexer file \"#1\" has length #2, incompatible with FrameSize #3");
         errMsg.replace(errMsg.find("#1"), 2, mFileStream->getFileName());
         errMsg.replace(errMsg.find("#2"), 2, std::to_string(fileSize));
         errMsg.replace(errMsg.find("#3"), 2, std::to_string(mFrameSize));
         throw std::invalid_argument(errMsg);
      }
   }
   else {
      mNumFrames = 0;
   }
   if (mFrameNumber > mNumFrames) {
      setFrameNumber(mNumFrames);
   }
}

long PVPFrameIndexer::calcFilePositionFromFrameNumber(long frameNumber) const {
   long filePos = mExternalHeaderSize + mFrameSize * frameNumber;
   return filePos;
}

int PVPFrameIndexer::calcFrameNumberFromFilePosition(long filePosition) const {
   long frameNumberL = (filePosition - mExternalHeaderSize) / mFrameSize;
   if (frameNumberL * mFrameSize + mExternalHeaderSize != filePosition) {
      std::string errMsg(
            "calcFrameNumberFromFilePosition() argument #1 is incompatible with "
            "external header size #2 and frame size #3");
      errMsg.replace(errMsg.find("#1"), 2, std::to_string(filePosition));
      errMsg.replace(errMsg.find("#2"), 2, std::to_string(mExternalHeaderSize));
      errMsg.replace(errMsg.find("#3"), 2, std::to_string(mFrameSize));
      throw std::invalid_argument(errMsg);
   }
   int frameNumber = static_cast<int>(frameNumberL);
   if (frameNumber != frameNumberL) {
      std::string errMsg(
            "calcFrameNumberFromFilePosition() argument #1 with external header size #2 and "
            "frame size #3 gives a frame number #4 that is larger than INT_MAX = %5");
      errMsg.replace(errMsg.find("#1"), 2, std::to_string(filePosition));
      errMsg.replace(errMsg.find("#2"), 2, std::to_string(mExternalHeaderSize));
      errMsg.replace(errMsg.find("#3"), 2, std::to_string(mFrameSize));
      errMsg.replace(errMsg.find("#4"), 2, std::to_string(frameNumberL));
      errMsg.replace(errMsg.find("#5"), 2, std::to_string(INT_MAX));
      throw std::invalid_argument(errMsg);
   }
   return frameNumber;
}

int PVPFrameIndexer::convertToLogicalFrameNumber(int frameNumber) {
   // A negative value means count from the end; for read-only a frameNumber
   // outside of usual limits means wrap around.
   int logicalFrameNumber = frameNumber;
   if (mReadOnlyFlag) {
      if (logicalFrameNumber < 0 or logicalFrameNumber >= mNumFrames) {
         logicalFrameNumber = logicalFrameNumber % mNumFrames;
      }
      if (logicalFrameNumber < 0) {
         logicalFrameNumber += mNumFrames;
      }
      pvAssert(logicalFrameNumber >= 0 and logicalFrameNumber < mNumFrames);
   }
   else {
      if (logicalFrameNumber < -mNumFrames or logicalFrameNumber > mNumFrames ) {
         std::string errMsg(
               "PVPFrameIndexer called with frame number #1 but "
               "file \"#2\" has only #3 frames");
         errMsg.replace(errMsg.find("#1"), 2, std::to_string(logicalFrameNumber));
         errMsg.replace(errMsg.find("#2"), 2, mFileStream->getFileName());
         errMsg.replace(errMsg.find("#3"), 2, std::to_string(mNumFrames));
         throw std::invalid_argument(errMsg);
      }
      if (logicalFrameNumber < 0) {
         logicalFrameNumber += mNumFrames;
      }
      pvAssert(logicalFrameNumber >= 0 and logicalFrameNumber <= mNumFrames);
   }
   return logicalFrameNumber;
}

void PVPFrameIndexer::moveFilePosToFrameStart() {
   long filePos = calcFilePositionFromFrameNumber(mFrameNumber);
   mFileStream->setInPos(filePos, std::ios_base::beg);
   if (!mReadOnlyFlag) {
      mFileStream->setOutPos(filePos, std::ios_base::beg);
   }
}

void PVPFrameIndexer::setFrameNumber(int frameNumber) {
   if (mFileStream == nullptr) { return; }
   initializeNumFrames();
   mFrameNumber = convertToLogicalFrameNumber(frameNumber);
   moveFilePosToFrameStart();
}

} // namespace PV
