#include "SharedWeightsFile.hpp"

#include "io/FileStreamBuilder.hpp"

namespace PV {

SharedWeightsFile::SharedWeightsFile(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &path,
      std::shared_ptr<WeightData> weightData,
      bool compressedFlag,
      bool readOnlyFlag,
      bool clobberFlag,
      bool verifyWrites)
      : WeightsFile(),
        mFileManager(fileManager),
        mPath(path),
        mPatchSizeX(weightData->getPatchSizeX()),
        mPatchSizeY(weightData->getPatchSizeY()),
        mPatchSizeF(weightData->getPatchSizeF()),
        mNumPatchesX(weightData->getNumDataPatchesX()),
        mNumPatchesY(weightData->getNumDataPatchesY()),
        mNumPatchesF(weightData->getNumDataPatchesF()),
        mNumArbors(weightData->getNumArbors()),
        mCompressedFlag(compressedFlag),
        mReadOnly(readOnlyFlag),
        mVerifyWrites(verifyWrites) {
   initializeCheckpointerDataInterface();
   initializeSharedWeightsIO(clobberFlag);
}

SharedWeightsFile::~SharedWeightsFile() {}

void SharedWeightsFile::read(WeightData &weightData) {
   double dummyTimestamp;
   readInternal(weightData, dummyTimestamp);
}

void SharedWeightsFile::read(WeightData &weightData, double &timestamp) {
   readInternal(weightData, timestamp);
   auto mpiComm = mFileManager->getMPIBlock()->getComm();
   MPI_Bcast(&timestamp, 1, MPI_DOUBLE, mFileManager->getRootProcessRank(), mpiComm);
}

void SharedWeightsFile::write(WeightData const &weightData, double timestamp) {
   if (isRoot()) {
      mSharedWeightsIO->write(weightData, timestamp);
   }
   setIndex(getIndex() + 1);
}

void SharedWeightsFile::truncate(int index) {
   FatalIf(
         mReadOnly,
         "SharedWeightsFile \"%s\" is read-only and cannot be truncated.\n",
         mPath.c_str());
   if (isRoot()) {
      int curFrameNumber  = mSharedWeightsIO->getFrameNumber();
      int lastFrameNumber = mSharedWeightsIO->getNumFrames();
      if (index >= lastFrameNumber) {
         WarnLog().printf(
               "Attempt to truncate \"%s\" to index %d, but file's max index is only %d\n",
               mPath.c_str(),
               index,
               lastFrameNumber);
         return;
      }
      int newFrameNumber = curFrameNumber > index ? index : curFrameNumber;
      long eofPosition   = mSharedWeightsIO->calcFilePositionFromFrameNumber(index);
      mSharedWeightsIO->close();
      mFileManager->truncate(mPath, eofPosition);
      mSharedWeightsIO->open();
      int newIndex = index < getIndex() ? index : getIndex();
      setIndex(newIndex);
   }
}

void SharedWeightsFile::setIndex(int index) {
   WeightsFile::setIndex(index);
   if (!isRoot()) {
      return;
   }
   int frameNumber = index;
   if (mReadOnly) {
      frameNumber = index % mSharedWeightsIO->getNumFrames();
   }
   mSharedWeightsIO->setFrameNumber(frameNumber);
   mFileStreamReadPos = mSharedWeightsIO->getFileStream()->getInPos();
   if (!mReadOnly) {
      mFileStreamWritePos = mSharedWeightsIO->getFileStream()->getOutPos();
   }
   else {
      mFileStreamWritePos = mFileStreamReadPos;
   }
}

Response::Status
SharedWeightsFile::registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
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

Response::Status SharedWeightsFile::processCheckpointRead(double simTime) {
   auto status = CheckpointerDataInterface::processCheckpointRead(simTime);
   if (!Response::completed(status)) {
      return status;
   }
   long pos  = mReadOnly ? mFileStreamReadPos : mFileStreamWritePos;
   int index = mSharedWeightsIO->calcFrameNumberFromFilePosition(pos);
   setIndex(index);
   if (isRoot() and mSharedWeightsIO->getFrameNumber() < mSharedWeightsIO->getNumFrames()) {
      WarnLog() << "Truncating \"" << getPath() << "\" to " << mSharedWeightsIO->getFrameNumber()
                << " frames.\n";
      truncate(getIndex());
   }
   return Response::SUCCESS;
}

int SharedWeightsFile::initializeCheckpointerDataInterface() {
   return CheckpointerDataInterface::initialize();
}

void SharedWeightsFile::initializeSharedWeightsIO(bool clobberFlag) {
   auto fileStream =
         FileStreamBuilder(
               mFileManager, mPath, false /*not text*/, mReadOnly, clobberFlag, mVerifyWrites)
               .get();

   mSharedWeightsIO = std::unique_ptr<SharedWeightsIO>(new SharedWeightsIO(
         fileStream,
         mPatchSizeX,
         mPatchSizeY,
         mPatchSizeF,
         mNumPatchesX,
         mNumPatchesY,
         mNumPatchesF,
         mNumArbors,
         mCompressedFlag));
}

void SharedWeightsFile::readInternal(WeightData &weightData, double &timestamp) {
   int status = PV_SUCCESS;
   if (isRoot()) {
      mSharedWeightsIO->read(weightData, timestamp);
   }
   int numElements = getPatchSizeOverall() * getNumPatchesOverall();
   int rootProc    = mFileManager->getRootProcessRank();
   auto mpiComm    = mFileManager->getMPIBlock()->getComm();
   for (int a = 0; a < getNumArbors(); ++a) {
      float *weightsData = weightData.getData(a);
      MPI_Bcast(weightsData, numElements, MPI_FLOAT, rootProc, mpiComm);
   }
   setIndex(getIndex() + 1);
}

} // namespace PV
