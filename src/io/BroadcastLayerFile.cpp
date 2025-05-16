#include "BroadcastLayerFile.hpp"

#include "checkpointing/CheckpointEntryFilePosition.hpp"
#include "io/FileStreamBuilder.hpp"

#include <algorithm> // std::copy
#include <memory>

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

Response::Status BroadcastLayerFile::processCheckpointRead(double simTime) {
   auto status = CheckpointerDataInterface::processCheckpointRead(simTime);
   if (!Response::completed(status)) {
      return status;
   }
   int index = mNumFrames / (mFileManager->getMPIBlock()->getBatchDimension() * mLocalBatchWidth);
   setIndex(index);
   if (mFileManager->isRoot() and mLayerIO->getFrameNumber() < mLayerIO->getNumFrames()) {
      WarnLog() << "Truncating \"" << getPath() << "\" to " << mLayerIO->getFrameNumber()
                << " frames.\n";
      truncate(mIndex);
   }
   return Response::SUCCESS;
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
      int blockBatchDimension = mpiBlock->getBatchDimension();
      int elementsPerBlock = mLocalBatchWidth * blockBatchDimension;
      for (int blockElement = 0; blockElement < elementsPerBlock; ++blockElement) {
         int mpiBatchIndex = blockElement / mLocalBatchWidth;
         int localBatchElement = blockElement % mLocalBatchWidth;
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
         float const *rootData = rootBuffer.asVector().data();
         for (int r = 0; r < mpiBlock->getNumRows(); ++r) {
            for (int c = 0; c < mpiBlock->getNumColumns(); ++c) {
               int rank = mpiBlock->calcRankFromRowColBatch(r, c, mpiBatchIndex);
               if (rank == mFileManager->getRootProcessRank()) {
                  std::copy(rootData, &rootData[mNumFeatures], getDataLocation(localBatchElement));
               }
               else {
                  MPI_Send(
                        rootData,
                        mNumFeatures,
                        MPI_FLOAT,
                        rank,
                        1731 + localBatchElement /*tag*/,
                        mpiBlock->getComm());
               }
            }
         }
      }
   }
   else {
      for (int b = 0; b < mLocalBatchWidth; ++b) {
         MPI_Recv(
               getDataLocation(b),
               mNumFeatures,
               MPI_FLOAT,
               mFileManager->getRootProcessRank(),
               1731 + b /*tag*/,
               mpiBlock->getComm(),
               MPI_STATUS_IGNORE);
      }
   }
   setIndex(mIndex + 1);
}

Response::Status
BroadcastLayerFile::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = CheckpointerDataInterface::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *checkpointer  = message->mDataRegistry;
   std::string dir     = dirName(mPath);
   std::string base    = stripExtension(mPath);
   std::string objName = dir + "/" + base;
   checkpointer->registerCheckpointData(
         objName,
         std::string("numframes"),
         &mNumFrames,
         (std::size_t)1,
         true /*broadcast*/,
         false /*not constant*/);
   auto filePosEntry = std::make_shared<CheckpointEntryFilePosition>(
         objName, std::string("filepos"), mLayerIO->getFileStream());
   bool registerSucceeded =
         checkpointer->registerCheckpointEntry(filePosEntry, false /*not constant for entire run*/);
   FatalIf(
         !registerSucceeded,
         "%s failed to register %s for checkpointing.\n",
         mPath.c_str(),
         filePosEntry->getName().c_str());
   return Response::SUCCESS;
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
      float *writeData = writeBuffer.asVector().data();
      int blockBatchDimension = mpiBlock->getBatchDimension();
      for (int m = 0; m < blockBatchDimension; ++m) {
         int sourceRank = mpiBlock->calcRankFromRowColBatch(0, 0, m);
         for (int b = 0; b < mLocalBatchWidth; ++b) {
            if (sourceRank == mpiBlock->getRank()) {
               float const *dataLocation = getDataLocation(b);
               std::copy(dataLocation, &dataLocation[mNumFeatures], writeData);
            }
            else {
               MPI_Recv(
                  writeData,
                  mNumFeatures,
                  MPI_FLOAT,
                  sourceRank,
                  1831 + b /*tag*/,
                  mpiBlock->getComm(),
                  MPI_STATUS_IGNORE);
            }
            mLayerIO->write(writeBuffer, timestamp);
         }
      }
   }
   else if (mpiBlock->getRowIndex() == 0 and mpiBlock->getColumnIndex() == 0) {
      // A broadcast layer has the same data across all rows and columns, so processes
      // with row index or column index nonzero does not have to do anything.
      int m = mpiBlock->getBatchIndex();
      for (int b = 0; b < mLocalBatchWidth; ++b) {
         MPI_Send(
            getDataLocation(b),
            mNumFeatures,
            MPI_FLOAT,
            mFileManager->getRootProcessRank(),
            1831 + b /*tag*/,
            mpiBlock->getComm());
      }
   }
   setIndex(mIndex + 1);
}

} // namespace PV
