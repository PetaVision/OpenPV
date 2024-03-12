#include "LayerFile.hpp"

#include "checkpointing/CheckpointEntryFilePosition.hpp"
#include "io/FileStreamBuilder.hpp"
#include "utils/BorderExchange.hpp"
#include "utils/PathComponents.hpp"

namespace PV {

LayerFile::LayerFile(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &path,
      PVLayerLoc const &layerLoc,
      bool dataExtendedFlag,
      bool fileExtendedFlag,
      bool readOnlyFlag,
      bool clobberFlag,
      bool verifyWrites)
      : CheckpointerDataInterface(),
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
   mDataLocations.resize(mLayerLoc.nbatch);
   initializeCheckpointerDataInterface();
   initializeGatherScatter();
   initializeLayerIO(clobberFlag);
}

void LayerFile::read() {
   double dummyTimestamp;
   readInternal(dummyTimestamp, false);
}

void LayerFile::read(double &timestamp) {
   readInternal(timestamp, true);
   auto mpiComm = mFileManager->getMPIBlock()->getComm();
   MPI_Bcast(&timestamp, 1, MPI_DOUBLE, mFileManager->getRootProcessRank(), mpiComm);
}

void LayerFile::write(double timestamp) {
   auto mpiBlock = mFileManager->getMPIBlock();
   if (isRoot()) {
      int nfRoot = mLayerLoc.nf;
      int nxRoot = mLayerLoc.nx * mpiBlock->getNumColumns();
      int nyRoot = mLayerLoc.ny * mpiBlock->getNumRows();
      if (mFileExtended) {
         nxRoot += mLayerLoc.halo.lt + mLayerLoc.halo.rt;
         nyRoot += mLayerLoc.halo.dn + mLayerLoc.halo.up;
      }
      Buffer<float> rootBuffer(nxRoot, nyRoot, nfRoot);
      for (int mpiBatchIndex = 0; mpiBatchIndex < mpiBlock->getBatchDimension(); ++mpiBatchIndex) {
         for (int b = 0; b < mLayerLoc.nbatch; ++b) {
            float *rootDataLocation = rootBuffer.asVector().data();
            mGatherScatter->gather(mpiBatchIndex, rootDataLocation, getDataLocation(b));
            mLayerIO->write(rootBuffer, timestamp);
         }
      }
   }
   else {
      for (int b = 0; b < mLayerLoc.nbatch; ++b) {
         mGatherScatter->gather(mpiBlock->getBatchIndex(), nullptr, getDataLocation(b));
      }
   }
   setIndex(mIndex + 1);
}

void LayerFile::truncate(int index) {
   FatalIf(mReadOnly, "LayerFile \"%s\" is read-only and cannot be truncated.\n", mPath.c_str());
   if (isRoot()) {
      int curFrameNumber    = mLayerIO->getFrameNumber();
      int lastFrameNumber   = mLayerIO->getNumFrames();
      int batchSize         = mFileManager->getMPIBlock()->getBatchDimension() * mLayerLoc.nbatch;
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
      int newIndex = index < getIndex() ? index : getIndex();
      setIndex(newIndex);
   }
}

void LayerFile::setIndex(int index) {
   mIndex = index;
   if (!isRoot()) {
      return;
   }
   int blockBatchDim = mFileManager->getMPIBlock()->getBatchDimension() * mLayerLoc.nbatch;
   int frameNumber   = index * blockBatchDim;
   if (mReadOnly) {
      frameNumber = frameNumber % mLayerIO->getNumFrames();
   }
   if (frameNumber < 0) {
      frameNumber += mLayerIO->getNumFrames();
   }
   if (frameNumber > mLayerIO->getNumFrames()) {
      int maxIndex = mLayerIO->getNumFrames() / blockBatchDim;
      Fatal().printf(
            "LayerFile::setIndex called for \"%s\" with index %d out of bounds. Allowed values for "
            "this file are 0 through %d (or -%d through 0, counting backwards from the end.)\n",
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

Response::Status
LayerFile::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
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

Response::Status LayerFile::processCheckpointRead(double simTime) {
   auto status = CheckpointerDataInterface::processCheckpointRead(simTime);
   if (!Response::completed(status)) {
      return status;
   }
   int index = mNumFrames / (mFileManager->getMPIBlock()->getBatchDimension() * mLayerLoc.nbatch);
   setIndex(index);
   if (isRoot() and mLayerIO->getFrameNumber() < mLayerIO->getNumFrames()) {
      WarnLog() << "Truncating \"" << getPath() << "\" to " << mLayerIO->getFrameNumber()
                << " frames.\n";
      truncate(mIndex);
   }
   return Response::SUCCESS;
}

int LayerFile::initializeCheckpointerDataInterface() {
   return CheckpointerDataInterface::initialize();
}

void LayerFile::initializeGatherScatter() {
   auto mpiBlock  = mFileManager->getMPIBlock();
   int rootRank   = mFileManager->getRootProcessRank();
   mGatherScatter = std::unique_ptr<LayerBatchGatherScatter>(
         new LayerBatchGatherScatter(mpiBlock, mLayerLoc, rootRank, mDataExtended, mFileExtended));
}

void LayerFile::initializeLayerIO(bool clobberFlag) {
   auto fileStream =
         FileStreamBuilder(
               mFileManager, mPath, false /*not text*/, mReadOnly, clobberFlag, mVerifyWrites)
               .get();

   auto mpiBlock = mFileManager->getMPIBlock();
   int nx        = mLayerLoc.nx * mpiBlock->getNumColumns();
   int ny        = mLayerLoc.ny * mpiBlock->getNumRows();
   int nf        = mLayerLoc.nf;
   if (mFileExtended) {
      nx += mLayerLoc.halo.lt + mLayerLoc.halo.rt;
      ny += mLayerLoc.halo.dn + mLayerLoc.halo.up;
   }
   mLayerIO = std::unique_ptr<LayerIO>(new LayerIO(fileStream, nx, ny, nf));
}

void LayerFile::readInternal(double &timestamp, bool checkTimestampConsistency) {
   auto mpiBlock = mFileManager->getMPIBlock();
   if (isRoot()) {
      bool checkTimestampActive = false; // becomes true after first read of a timestamp
      int nfRoot                = mLayerLoc.nf;
      int nxRoot                = mLayerLoc.nx * mpiBlock->getNumColumns();
      int nyRoot                = mLayerLoc.ny * mpiBlock->getNumRows();
      if (mFileExtended) {
         nxRoot += mLayerLoc.halo.lt + mLayerLoc.halo.rt;
         nyRoot += mLayerLoc.halo.up + mLayerLoc.halo.dn;
      }
      Buffer<float> rootBuffer(nxRoot, nyRoot, nfRoot);
      int batchDimension = mpiBlock->getBatchDimension();
      for (int mpiBatchIndex = 0; mpiBatchIndex < batchDimension; ++mpiBatchIndex) {
         for (int b = 0; b < mLayerLoc.nbatch; ++b) {
            double thisTimestamp;
            mLayerIO->read(rootBuffer, thisTimestamp);
            if (checkTimestampActive and thisTimestamp != timestamp) {
               WarnLog() << "LayerFile::read() frame timestamps are inconsistent\n";
            }
            checkTimestampActive = checkTimestampConsistency;
            // If we don't care about the timestamp, checkTimestampActive never becomes true and
            // the warning above is never triggered.
            timestamp = thisTimestamp;
            if (mReadOnly and mLayerIO->getFrameNumber() == mLayerIO->getNumFrames()) {
               mLayerIO->setFrameNumber(0);
            }
            float *rootDataLocation = rootBuffer.asVector().data();
            mGatherScatter->scatter(mpiBatchIndex, rootDataLocation, getDataLocation(b));
         }
      }
   }
   else {
      for (int b = 0; b < mLayerLoc.nbatch; ++b) {
         mGatherScatter->scatter(mpiBlock->getBatchIndex(), nullptr, getDataLocation(b));
      }
   }

   auto localMPIBlock  = mFileManager->getMPIBlock();
   auto globalMPIBlock = MPIBlock(
         localMPIBlock->getGlobalComm(),
         localMPIBlock->getGlobalNumRows(),
         localMPIBlock->getGlobalNumColumns(),
         localMPIBlock->getGlobalBatchDimension(),
         localMPIBlock->getGlobalNumRows(),
         localMPIBlock->getGlobalNumColumns(),
         localMPIBlock->getGlobalBatchDimension());
   auto borderExchange = BorderExchange(globalMPIBlock, mLayerLoc);
   std::vector<MPI_Request> requestsVector;
   for (int b = 0; b < mLayerLoc.nbatch; b++) {
      float *data = getDataLocation(b);
      std::vector<MPI_Request> batchElementMPIRequest;
      borderExchange.exchange(data, batchElementMPIRequest);
      requestsVector.insert(
            requestsVector.end(), batchElementMPIRequest.begin(), batchElementMPIRequest.end());
   }
   borderExchange.wait(requestsVector);

   setIndex(mIndex + 1);
}

LayerFile::~LayerFile() {}

} // namespace PV
