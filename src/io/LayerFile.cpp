#include "LayerFile.hpp"

#include "io/FileStreamBuilder.hpp"

namespace PV {

LayerFile::LayerFile(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &path,
      PVLayerLoc const &layerLoc,
      bool dataExtendedFlag,
      bool fileExtendedFlag,
      bool readOnlyFlag,
      bool verifyWrites) :
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
   int mpiBatchDimension = mFileManager->getMPIBlock()->getBatchDimension();
   mDataLocations.resize(mLayerLoc.nbatch);
   initializeGatherScatter();
   initializeLayerIO();
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
      int curFrameNumber = mLayerIO->getFrameNumber();
      int lastFrameNumber = mLayerIO->getNumFrames();
      int batchSize = mFileManager->getMPIBlock()->getBatchDimension() * mLayerLoc.nbatch;
      int targetFrameNumber = index * batchSize;
      if (targetFrameNumber >= lastFrameNumber) {
         WarnLog().printf(
               "Attempt to truncate \"%s\" to index %d, but file's max index is only %d\n",
               mPath.c_str(), index, lastFrameNumber / batchSize);
         return;
      }
      int newFrameNumber = curFrameNumber > targetFrameNumber ? targetFrameNumber : curFrameNumber;
      long eofPosition = mLayerIO->calcFilePositionFromFrameNumber(newFrameNumber);
      mLayerIO = std::unique_ptr<LayerIO>(); // closes existing file
      mFileManager->truncate(mPath, eofPosition);
      initializeLayerIO(); // reopens existing file with same mode.
      mLayerIO->setFrameNumber(newFrameNumber);
   }
}

void LayerFile::setIndex(int index) {
   mIndex = index;
   if (!isRoot()) { return; }
   int frameNumber = index * mFileManager->getMPIBlock()->getBatchDimension() * mLayerLoc.nbatch;
   if (mReadOnly) {
      frameNumber = frameNumber % mLayerIO->getNumFrames();
   }
   mLayerIO->setFrameNumber(frameNumber);
}

void LayerFile::initializeGatherScatter() {
   auto mpiBlock = mFileManager->getMPIBlock();
   int rootRank = mFileManager->getRootProcessRank();
   mGatherScatter = std::unique_ptr<LayerBatchGatherScatter>(new LayerBatchGatherScatter(
         mpiBlock, mLayerLoc, rootRank, mDataExtended, mFileExtended));
}

void LayerFile::initializeLayerIO() {
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
   mLayerIO = std::unique_ptr<LayerIO>(new LayerIO(fileStream, nx, ny, nf));
}

void LayerFile::readInternal(double &timestamp, bool checkTimestampConsistency) {
   auto mpiBlock = mFileManager->getMPIBlock();
   if (isRoot()) {
      bool checkTimestampActive = false; // becomes true after first read of a timestamp
      int nfRoot = mLayerLoc.nf;
      int nxRoot = mLayerLoc.nx * mpiBlock->getNumColumns();
      int nyRoot = mLayerLoc.ny * mpiBlock->getNumRows();
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
   setIndex(mIndex + 1);
}

LayerFile::~LayerFile() {}

} // namespace PV
