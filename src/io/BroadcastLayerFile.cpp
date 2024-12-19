#include "BroadcastLayerFile.hpp"

#include "io/FileStreamBuilder.hpp"

#include <algorithm> // std::copy

namespace PV {

BroadcastLayerFile::BroadcastLayerFile(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &path,
      int numFeatures,
      int localBatchWidth,
      bool readOnlyFlag,
      bool clobberFlag,
      bool verifyWrites)
      : CheckpointerDataInterface(),
        mFileManager(fileManager),
        mPath(path),
        mNumFeatures(numFeatures),
        mLocalBatchWidth(localBatchWidth),
        mReadOnly(readOnlyFlag),
        mVerifyWrites(verifyWrites) {
   CheckpointerDataInterface::initialize();
   mDataLocations.resize(localBatchWidth);
   initializeLayerIO(clobberFlag);
}

BroadcastLayerFile::~BroadcastLayerFile() {}

void BroadcastLayerFile::initializeLayerIO(bool clobberFlag) {
   auto fileStream =
         FileStreamBuilder(
               mFileManager, mPath, false /*not text*/, mReadOnly, clobberFlag, mVerifyWrites)
               .get();

   mLayerIO = std::unique_ptr<LayerIO>(new LayerIO(fileStream, 1 /*nx*/, 1 /*ny*/, mNumFeatures));
}

void BroadcastLayerFile::read() {
   double dummyTimestamp;
   readInternal(dummyTimestamp, false);
}

void BroadcastLayerFile::read(double &timestamp) {
   readInternal(timestamp, true);
   auto mpiComm = mFileManager->getMPIBlock()->getComm();
   MPI_Bcast(&timestamp, 1, MPI_DOUBLE, mFileManager->getRootProcessRank(), mpiComm);
}

void BroadcastLayerFile::readInternal(double &timestamp, bool checkTimestampConsistency) {
   auto mpiBlock = mFileManager->getMPIBlock();
   if (mFileManager->isRoot()) {
      bool checkTimestampActive = false;
      // If checkTimestampConsistency is set, checkTimestampActive becomes true after first read
      Buffer<float> rootBuffer(1, 1, mNumFeatures);
      int mpiBatchDimension = mpiBlock->getBatchDimension();
      for (int mpiBatchIndex = 0; mpiBatchIndex < mpiBatchDimension; ++mpiBatchIndex) {
         for (int b = 0; b < mLocalBatchWidth; ++b) {
            double thisTimestamp;
            mLayerIO->read(rootBuffer, thisTimestamp);
            if (checkTimestampActive and thisTimestamp != timestamp) {
               WarnLog() << "BroadcastLayerFile::read() frame timestamps are inconsistent\n";
            }
            checkTimestampActive = checkTimestampConsistency;
            // If we don't care about the timestamp, checkTimestampActive never becomes true and
            // the warning above is never triggered.
            timestamp = thisTimestamp;
            if (mReadOnly and mLayerIO->getFrameNumber() == mLayerIO->getNumFrames()) {
               mLayerIO->setFrameNumber(0);
            }
            float const *rootDataLocation = rootBuffer.asVector().data();
            std::copy(rootDataLocation, &rootDataLocation[mNumFeatures], getDataLocation(b));
            MPI_Bcast(
                  getDataLocation(b),
                  mNumFeatures,
                  MPI_FLOAT,
                  mFileManager->getRootProcessRank(),
                  mpiBlock->getComm());
         }
      }
   }
   else {
      for (int b = 0; b < mLocalBatchWidth; ++b) {
         MPI_Bcast(
               getDataLocation(b),
               mNumFeatures,
               MPI_FLOAT,
               mFileManager->getRootProcessRank(),
               mpiBlock->getComm());
      }
   }

   setIndex(mIndex + 1);
}

void BroadcastLayerFile::setIndex(int index) {
   mIndex = index;
   if (!mFileManager->isRoot()) {
      return;
   }
   int blockBatchDim = mFileManager->getMPIBlock()->getBatchDimension() * mLocalBatchWidth;
   int frameNumber   = index * blockBatchDim;
   if (mReadOnly) {
      FatalIf(
            mLayerIO->getNumFrames() == 0,
            "Read-only file \"%s\" has zero frames; cannot set index to %d\n",
            mPath.c_str(),
            index);
      frameNumber = frameNumber % mLayerIO->getNumFrames();
   }
   if (frameNumber < 0) {
      frameNumber += mLayerIO->getNumFrames();
   }
   if (frameNumber > mLayerIO->getNumFrames()) {
      int maxIndex = mLayerIO->getNumFrames() / blockBatchDim;
      Fatal().printf(
            "BroadcastLayerFile::setIndex called for \"%s\" with index %d out of bounds. "
            "Allowed values for this file are 0 through %d (or -%d through 0, counting backwards "
            "from the end.)\n",
            mFileManager->makeBlockFilename(getPath()).c_str(),
            index,
            maxIndex,
            maxIndex);
   }
   mLayerIO->setFrameNumber(frameNumber);
   mNumFrames         = frameNumber;
   mFileStreamReadPos = mLayerIO->getFileStream()->getInPos();
   if (!mReadOnly) {
      mFileStreamWritePos = mLayerIO->getFileStream()->getOutPos();
   }
   else {
      mFileStreamWritePos = mFileStreamReadPos;
   }
}

void BroadcastLayerFile::truncate(int index) {
   FatalIf(
         mReadOnly,
         "BroadcastLayerFile \"%s\" is read-only and cannot be truncated.\n",
         mPath.c_str());
   if (mFileManager->isRoot()) {
      int curFrameNumber    = mLayerIO->getFrameNumber();
      int lastFrameNumber   = mLayerIO->getNumFrames();
      int batchSize         = mFileManager->getMPIBlock()->getBatchDimension() * mLocalBatchWidth;
      int targetFrameNumber = index * batchSize;
      if (targetFrameNumber >= lastFrameNumber) {
         WarnLog().printf(
               "Attempt to truncate \"%s\" to index %d, but file's max index is only %d\n",
               mPath.c_str(),
               index,
               lastFrameNumber / batchSize);
         return;
      }
      int newFrameNumber = curFrameNumber > targetFrameNumber ? targetFrameNumber : curFrameNumber;
      long eofPosition   = mLayerIO->calcFilePositionFromFrameNumber(newFrameNumber);
      mLayerIO->close();
      mFileManager->truncate(mPath, eofPosition);
      mLayerIO->open();
   }
   int newIndex = index < getIndex() ? index : getIndex();
   setIndex(newIndex);
}

void BroadcastLayerFile::write(double timestamp) {
   auto mpiBlock = mFileManager->getMPIBlock();
   if (mFileManager->isRoot()) {
      Buffer<float> writeBuffer(1 /*nx*/, 1 /*ny*/, mNumFeatures);
      for (int mpiBatchIndex = 0; mpiBatchIndex < mpiBlock->getBatchDimension(); ++mpiBatchIndex) {
         for (int b = 0; b < mLocalBatchWidth; ++b) {
            float *writeBufferLocation = writeBuffer.asVector().data();
            float const *layerData = getDataLocation(b);
            std::copy(layerData, &layerData[mNumFeatures], writeBufferLocation);
            mLayerIO->write(writeBuffer, timestamp);
         }
      }
   }
   // No need for nonroot processes to do anything since all processes have the same data in a
   // broadcast layer
   setIndex(mIndex + 1);
}

} // namespace PV
