#include "SparseLayerFile.hpp"

#include "io/FileStreamBuilder.hpp"

namespace PV {

SparseLayerFile::SparseLayerFile(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &path,
      PVLayerLoc const &layerLoc,
      bool dataExtendedFlag,
      bool fileExtendedFlag,
      bool readOnlyFlag,
      bool verifyWrites) :
      CheckpointerDataInterface(),
      mFileManager(fileManager),
      mPath(path),
      mLayerLoc(layerLoc),
      mDataExtended(dataExtendedFlag),
      mFileExtended(fileExtendedFlag),
      mReadOnly(readOnlyFlag),
      mVerifyWrites(verifyWrites) {

   if (!mDataExtended) {
      mLayerLoc.halo.lt = 0;
      mLayerLoc.halo.rt = 0;
      mLayerLoc.halo.dn = 0;
      mLayerLoc.halo.up = 0;
   }
   mSparseListLocations.resize(mLayerLoc.nbatch);
   initializeCheckpointerDataInterface();
   initializeGatherScatter();
   initializeSparseLayerIO();
}

SparseLayerFile::~SparseLayerFile() {}

void SparseLayerFile::read() {
   double dummyTimestamp;
   readInternal(dummyTimestamp, false);
}

void SparseLayerFile::read(double &timestamp) {
   readInternal(timestamp, true);
}

void SparseLayerFile::write(double timestamp) {
   if (isRoot()) {
      if (mSparseLayerIO->getFrameNumber() < mSparseLayerIO->getNumFrames()) {
         WarnLog() << "Truncating \"" << getPath() << "\" to "
                   << mSparseLayerIO->getFrameNumber() << " frames.\n";
         truncate(mIndex);
      }
      SparseList<float> rootSparseList;
      int mpiBatchDimension = mFileManager->getMPIBlock()->getBatchDimension();
      for (int mpiBatchIndex = 0; mpiBatchIndex < mpiBatchDimension; ++mpiBatchIndex) {
         for (int b = 0; b < mLayerLoc.nbatch; ++b) {
            mGatherScatter->gather(mpiBatchIndex, &rootSparseList, getListLocation(b));
            mSparseLayerIO->write(rootSparseList, timestamp);
         }
      }
   }
   else {
      for (int b = 0; b < mLayerLoc.nbatch; ++b) {
         int batchIndex = mFileManager->getMPIBlock()->getBatchIndex();
         mGatherScatter->gather(batchIndex, nullptr, getListLocation(b));
      }
   }
   setIndex(mIndex + 1);
}

void SparseLayerFile::truncate(int index) {
   FatalIf(
         mReadOnly,
         "SparseLayerFile \"%s\" is read-only and cannot be truncated.\n",
         mPath.c_str());
   if (isRoot()) {
      int curFrameNumber = mSparseLayerIO->getFrameNumber();
      int lastFrameNumber = mSparseLayerIO->getNumFrames();
      int batchSize = mFileManager->getMPIBlock()->getBatchDimension() * mLayerLoc.nbatch;
      int targetFrameNumber = index * batchSize;
      if (targetFrameNumber >= lastFrameNumber) {
         WarnLog().printf(
               "Attempt to truncate \"%s\" to index %d, but file's max index is only %d\n",
               mPath.c_str(), index, lastFrameNumber / batchSize);
         return;
      }
      int newFrameNumber = curFrameNumber > targetFrameNumber ? targetFrameNumber : curFrameNumber;
      long filePosition = mSparseLayerIO->calcFilePositionFromFrameNumber(newFrameNumber);
      mSparseLayerIO = std::unique_ptr<SparseLayerIO>(); // closes existing file
      mFileManager->truncate(mPath, filePosition);
      initializeSparseLayerIO(); // reopens existing file with same mode.
      mSparseLayerIO->setFrameNumber(newFrameNumber);
   }
}

Response::Status
SparseLayerFile::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
   auto status = CheckpointerDataInterface::registerData(message);
   if (!Response::completed(status)) {
      return status;
   }
   auto *checkpointer = message->mDataRegistry;
   std::string dir  = dirName(mPath);
   std::string base = stripExtension(mPath);
   std::string objName = dir + "/" + base;
   checkpointer->registerCheckpointData(
         objName,
         std::string("numframes_sparse"),
         &mNumFramesSparse,
         (std::size_t)1,
         true /*broadcast*/,
         false /*not constant*/);
   checkpointer->registerCheckpointData(
         objName,
         std::string("filepos_FileStreamRead"),
         &mFileStreamReadPos,
         (std::size_t)1,
         true /*broadcast*/,
         false /*not constant*/);
   checkpointer->registerCheckpointData(
         objName,
         std::string("filepos_FileStreamWrite"),
         &mFileStreamWritePos,
         (std::size_t)1,
         true /*broadcast*/,
         false /*not constant*/);
   return Response::SUCCESS;
}

void SparseLayerFile::setIndex(int index) {
   mIndex = index;
   if (!isRoot()) { return; }
   int frameNumber = index * mFileManager->getMPIBlock()->getBatchDimension() * mLayerLoc.nbatch;
   if (mReadOnly) {
      frameNumber = frameNumber % mSparseLayerIO->getNumFrames();
   }
   mSparseLayerIO->setFrameNumber(frameNumber);
}

Response::Status SparseLayerFile::processCheckpointRead() {
   auto status = CheckpointerDataInterface::processCheckpointRead();
   if (!Response::completed(status)) {
      return status;
   }
   int index =
         mNumFramesSparse / (mFileManager->getMPIBlock()->getBatchDimension() * mLayerLoc.nbatch);
   setIndex(index);
   return Response::SUCCESS;
}

int SparseLayerFile::initializeCheckpointerDataInterface() {
   return CheckpointerDataInterface::initialize();
}

void SparseLayerFile::initializeGatherScatter() {
   mGatherScatter = std::unique_ptr<SparseLayerBatchGatherScatter>(
         new SparseLayerBatchGatherScatter(
               mFileManager->getMPIBlock(),
               mLayerLoc,
               mFileManager->getRootProcessRank(),
               mDataExtended,
               mFileExtended));
}

void SparseLayerFile::initializeSparseLayerIO() {
   auto fileStream = FileStreamBuilder(
         mFileManager, mPath, false /* not text */, mReadOnly, mVerifyWrites).get();

   auto mpiBlock = mFileManager->getMPIBlock();
   int nx = mLayerLoc.nx * mpiBlock->getNumColumns();
   int ny = mLayerLoc.ny * mpiBlock->getNumRows();
   int nf = mLayerLoc.nf;
   if (mFileExtended) {
      nx += mLayerLoc.halo.lt + mLayerLoc.halo.rt;
      ny += mLayerLoc.halo.dn + mLayerLoc.halo.up;
   }
   mSparseLayerIO = std::unique_ptr<SparseLayerIO>(new SparseLayerIO(fileStream, nx, ny, nf));
}

void SparseLayerFile::readInternal(double &timestamp, bool checkTimestampConsistency) {
   auto mpiBlock = mFileManager->getMPIBlock();
   SparseList<float> localList;
   if (isRoot()) {
      bool checkTimestampActive = false; // becomes true after first read of a timestamp
      int nfRoot = mLayerLoc.nf;
      int nxRoot = mLayerLoc.nx * mpiBlock->getNumColumns();
      int nyRoot = mLayerLoc.ny * mpiBlock->getNumRows();
      if (mFileExtended) {
         nxRoot += mLayerLoc.halo.lt + mLayerLoc.halo.rt;
         nyRoot += mLayerLoc.halo.up + mLayerLoc.halo.dn;
      }
      SparseList<float> rootSparseList(nxRoot, nyRoot, nfRoot);
      for (int mpiBatchIndex = 0; mpiBatchIndex < mpiBlock->getBatchDimension(); ++mpiBatchIndex) {
         for (int b = 0; b < mLayerLoc.nbatch; ++b) {
            double thisTimestamp;
            mSparseLayerIO->read(rootSparseList, thisTimestamp);
            if (checkTimestampActive and thisTimestamp != timestamp) {
                WarnLog() << "SparseLayerFile::read() frame timestamps are inconsistent\n";
            }
            checkTimestampActive = checkTimestampConsistency;
            // If we don't care about the timestamp, checkTimestampActive never becomes true and
            // the warning above is never triggered.
            timestamp = thisTimestamp;
            if (mReadOnly and mSparseLayerIO->getFrameNumber() == mSparseLayerIO->getNumFrames()) {
               mSparseLayerIO->setFrameNumber(0);
            }
            mGatherScatter->scatter(mpiBatchIndex, &rootSparseList, getListLocation(b));
         }
      }
   }
   else {
      for (int b = 0; b < mLayerLoc.nbatch; ++b) {
         mGatherScatter->scatter(mpiBlock->getBatchIndex(), nullptr, getListLocation(b));
      }
   }
   setIndex(mIndex + 1);
}

} // namespace PV
