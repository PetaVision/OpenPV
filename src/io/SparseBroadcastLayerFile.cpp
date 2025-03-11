#include "SparseBroadcastLayerFile.hpp"

#include "checkpointing/CheckpointEntryFilePosition.hpp"
#include "io/FileStreamBuilder.hpp"

namespace PV {

SparseBroadcastLayerFile::SparseBroadcastLayerFile(
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
   mSparseListLocations.resize(mLocalBatchWidth);
   initializeSparseLayerIO(clobberFlag);
}

SparseBroadcastLayerFile::~SparseBroadcastLayerFile() {}

void SparseBroadcastLayerFile::read() {
   double dummyTimestamp;
   readInternal(dummyTimestamp, false);
}

void SparseBroadcastLayerFile::read(double &timestamp) {
   readInternal(timestamp, true);
   auto mpiComm = mFileManager->getMPIBlock()->getComm();
   MPI_Bcast(&timestamp, 1, MPI_DOUBLE, mFileManager->getRootProcessRank(), mpiComm);
}

void SparseBroadcastLayerFile::write(double timestamp) {
   if (isRoot()) {
      // TODO: If writing to the middle of the file, move the tail end of the
      // file as necessary, instead of truncating. This is necessary for
      // sparse layer files but not for other PVP file classes because
      // frames in sparse format have variable sizes.
      if (mSparseLayerIO->getFrameNumber() < mSparseLayerIO->getNumFrames()) {
         WarnLog() << "Truncating \"" << getPath() << "\" to " << mSparseLayerIO->getFrameNumber()
                   << " frames.\n";
         truncate(mIndex);
      }
      SparseList<float> rootSparseList;
      int mpiBatchDimension = mFileManager->getMPIBlock()->getBatchDimension();
      for (int mpiBatchIndex = 0; mpiBatchIndex < mpiBatchDimension; ++mpiBatchIndex) {
         for (int b = 0; b < mLocalBatchWidth; ++b) {
            gather(mpiBatchIndex, b, &rootSparseList);
            mSparseLayerIO->write(rootSparseList, timestamp);
         }
      }
   }
   else {
      for (int b = 0; b < mLocalBatchWidth; ++b) {
         int batchIndex = mFileManager->getMPIBlock()->getBatchIndex();
         gather(batchIndex, b, nullptr);
      }
   }
   setIndex(mIndex + 1);
}

void SparseBroadcastLayerFile::truncate(int index) {
   FatalIf(
         mReadOnly,
         "SparseBroadcastLayerFile \"%s\" is read-only and cannot be truncated.\n",
         mPath.c_str());
   if (isRoot()) {
      int curFrameNumber    = mSparseLayerIO->getFrameNumber();
      int lastFrameNumber   = mSparseLayerIO->getNumFrames();
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
      long filePosition  = mSparseLayerIO->calcFilePositionFromFrameNumber(newFrameNumber);
      mSparseLayerIO->close();
      mFileManager->truncate(mPath, filePosition);
      mSparseLayerIO->open();
      int newIndex = index < getIndex() ? index : getIndex();
      setIndex(newIndex);
   }
}

Response::Status
SparseBroadcastLayerFile::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
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
         std::string("numframes_sparse"),
         &mNumFramesSparse,
         (std::size_t)1,
         true /*broadcast*/,
         false /*not constant*/);
   auto filePosEntry = std::make_shared<CheckpointEntryFilePosition>(
         objName, std::string("filepos"), mSparseLayerIO->getFileStream());
   bool registerSucceeded =
         checkpointer->registerCheckpointEntry(filePosEntry, false /*not constant for entire run*/);
   FatalIf(
         !registerSucceeded,
         "%s failed to register %s for checkpointing.\n",
         mPath.c_str(),
         filePosEntry->getName().c_str());
   return Response::SUCCESS;
}

void SparseBroadcastLayerFile::setIndex(int index) {
   mIndex = index;
   if (!isRoot()) {
      return;
   }
   int blockBatchDim = mFileManager->getMPIBlock()->getBatchDimension() * mLocalBatchWidth;
   int frameNumber   = index * blockBatchDim;
   if (mReadOnly) {
      FatalIf(
            mSparseLayerIO->getNumFrames() == 0,
            "Read-only file \"%s\" has zero frames; cannot set index to %d\n",
            mPath.c_str(),
            index);
      frameNumber = frameNumber % mSparseLayerIO->getNumFrames();
   }
   if (frameNumber < 0) {
      frameNumber += mSparseLayerIO->getNumFrames();
   }
   if (frameNumber > mSparseLayerIO->getNumFrames()) {
      int maxIndex = mSparseLayerIO->getNumFrames() / blockBatchDim;
      Fatal().printf(
            "SparseBroadcastLayerFile::setIndex called for \"%s\" with index %d out of bounds. "
            "Allowed values for this file are 0 through %d (or -%d through 0, counting backwards "
            "from the end.)\n",
            mFileManager->makeBlockFilename(getPath()).c_str(),
            index,
            maxIndex,
            maxIndex);
   }
   mSparseLayerIO->setFrameNumber(frameNumber);
   mNumFramesSparse   = frameNumber;
   mFileStreamReadPos = mSparseLayerIO->getFileStream()->getInPos();
   if (!mReadOnly) {
      mFileStreamWritePos = mSparseLayerIO->getFileStream()->getOutPos();
   }
   else {
      mFileStreamWritePos = mFileStreamReadPos;
   }
}

Response::Status SparseBroadcastLayerFile::processCheckpointRead(double simTime) {
   auto status = CheckpointerDataInterface::processCheckpointRead(simTime);
   if (!Response::completed(status)) {
      return status;
   }
   int index =
         mNumFramesSparse / (mFileManager->getMPIBlock()->getBatchDimension() * mLocalBatchWidth);
   setIndex(index);
   if (isRoot() and mSparseLayerIO->getFrameNumber() < mSparseLayerIO->getNumFrames()) {
      WarnLog() << "Truncating \"" << getPath() << "\" to " << mSparseLayerIO->getFrameNumber()
                << " frames.\n";
      truncate(mIndex);
   }
   return Response::SUCCESS;
}

void SparseBroadcastLayerFile::initializeSparseLayerIO(bool clobberFlag) {
   auto fileStream =
         FileStreamBuilder(
               mFileManager, mPath, false /*not text*/, mReadOnly, clobberFlag, mVerifyWrites)
               .get();

   mSparseLayerIO = std::unique_ptr<SparseLayerIO>(
         new SparseLayerIO(fileStream, 1/*nx*/, 1/*ny*/, mNumFeatures));
}

void SparseBroadcastLayerFile::readInternal(double &timestamp, bool checkTimestampConsistency) {
   auto mpiBlock = mFileManager->getMPIBlock();
   SparseList<float> localList;
   if (isRoot()) {
      bool checkTimestampActive = false; // becomes true after first read of a timestamp
      SparseList<float> sparseListFromFile(1, 1, mNumFeatures);
      for (int mpiBatchIndex = 0; mpiBatchIndex < mpiBlock->getBatchDimension(); ++mpiBatchIndex) {
         for (int b = 0; b < mLocalBatchWidth; ++b) {
            double thisTimestamp;
            mSparseLayerIO->read(sparseListFromFile, thisTimestamp);
            if (checkTimestampActive and thisTimestamp != timestamp) {
               WarnLog() << "SparseBroadcastLayerFile::read() frame timestamps are inconsistent\n";
            }
            checkTimestampActive = checkTimestampConsistency;
            // If we don't care about the timestamp, checkTimestampActive never becomes true and
            // the warning above is never triggered.
            timestamp = thisTimestamp;
            if (mReadOnly and mSparseLayerIO->getFrameNumber() == mSparseLayerIO->getNumFrames()) {
               mSparseLayerIO->setFrameNumber(0);
            }
            scatter(mpiBatchIndex, b, &sparseListFromFile);
         }
      }
   }
   else {
      for (int b = 0; b < mLocalBatchWidth; ++b) {
         scatter(mpiBlock->getBatchIndex(), b, nullptr);
      }
   }
   setIndex(mIndex + 1);
}

void SparseBroadcastLayerFile::gather(
      int mpiBatchIndex,
      int localBatchIndex,
      SparseList<float> *rootSparseList) {
   int rootProc = 0;
   int tag = 3715 + localBatchIndex;
   SparseList<float> const *localSparseList = getListLocation(localBatchIndex);
   auto mpiBlock = mFileManager->getMPIBlock();
   if (isRoot()) {
      rootSparseList->reset(1, 1, localSparseList->getFeatures());
      int sourceRank = mpiBlock->calcRankFromRowColBatch(0, 0, mpiBatchIndex);
      if (sourceRank == rootProc) {
         rootSparseList->set(localSparseList->getContents());
      }
      else {
         MPI_Status mpiStatus;
         MPI_Probe(sourceRank, tag, mpiBlock->getComm(), &mpiStatus);
         int count;
         MPI_Get_count(&mpiStatus, MPI_BYTE, &count);
         int sizePerEntry = static_cast<int>(sizeof(SparseList<float>::Entry));
         FatalIf(
               count % sizePerEntry != 0,
               "SparseBroadcastLayerFile::scatter() receiving %d bytes, "
               "which is not a multiple of sparse entry size of %d bytes.\n",
               count, sizePerEntry);
         int numEntries = count / sizePerEntry;
         std::vector<SparseList<float>::Entry> sparseContents(numEntries);
         MPI_Recv(
               sparseContents.data(),
               count,
               MPI_BYTE,
               rootProc,
               tag,
               mpiBlock->getComm(),
               MPI_STATUS_IGNORE);
         rootSparseList->set(sparseContents);
      }
   }
   else {
      bool batchMatch = mpiBlock->getBatchIndex() == mpiBatchIndex;
      bool baseRowColumn = mpiBlock->getRowIndex() == 0 and mpiBlock->getColumnIndex() == 0;
      if (batchMatch and baseRowColumn) {
         auto sparseContents = localSparseList->getContents();
         MPI_Send(
               sparseContents.data(),
               static_cast<int>(sizeof(SparseList<float>::Entry) * sparseContents.size()),
               MPI_BYTE,
               rootProc,
               tag,
               mpiBlock->getComm());
      }
   }
}

void SparseBroadcastLayerFile::scatter(
      int mpiBatchIndex,
      int localBatchIndex,
      SparseList<float> const *rootSparseList) {
   int rootProc = 0;
   int tag = 3715 + localBatchIndex;
   SparseList<float> *localSparseList = getListLocation(localBatchIndex);
   auto mpiBlock = mFileManager->getMPIBlock();
   if (isRoot()) {
      int numCols = mpiBlock->getNumColumns();
      int numRows = mpiBlock->getNumRows();
      for (int row = 0; row < numRows; ++row) {
         for (int col = 0; col < numCols; ++col) {
            int destRank = mpiBlock->calcRankFromRowColBatch(row, col, mpiBatchIndex);
            auto sparseContents = rootSparseList->getContents();
            if (destRank == rootProc) {
               localSparseList->set(sparseContents);
            }
            else {
               MPI_Send(
                     sparseContents.data(),
                     static_cast<int>(sizeof(SparseList<float>::Entry) * sparseContents.size()),
                     MPI_BYTE,
                     destRank,
                     tag,
                     mpiBlock->getComm());
            }
         }
      }
   }
   else {
      MPI_Status mpiStatus;
      MPI_Probe(rootProc, tag, mpiBlock->getComm(), &mpiStatus);
      int count;
      MPI_Get_count(&mpiStatus, MPI_BYTE, &count);
      int sizePerEntry = static_cast<int>(sizeof(SparseList<float>::Entry));
      FatalIf(
            count % sizePerEntry != 0,
            "SparseBroadcastLayerFile::scatter() receiving %d bytes, "
            "which is not a multiple of sparse entry size of %d bytes.\n",
            count, sizePerEntry);
      int numEntries = count / sizePerEntry;
      std::vector<SparseList<float>::Entry> sparseContents(numEntries);
      MPI_Recv(
            sparseContents.data(),
            count,
            MPI_BYTE,
            rootProc,
            tag,
            mpiBlock->getComm(),
            MPI_STATUS_IGNORE);
      localSparseList->set(sparseContents);
   }
}

} // namespace PV
