#include "SharedWeightsFile.hpp"

#include "io/FileStreamBuilder.hpp"

namespace PV {

SharedWeightsFile::SharedWeightsFile(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &path,
      int patchSizeX,
      int patchSizeY,
      int patchSizeF,
      long numPatchesX,
      long numPatchesY,
      long numPatchesF,
      int numArbors,
      bool compressedFlag,
      bool readOnlyFlag,
      bool verifyWrites) :
      mFileManager(fileManager),
      mPath(path),
      mPatchSizeX(patchSizeX),
      mPatchSizeY(patchSizeY),
      mPatchSizeF(patchSizeF),
      mNumPatchesX(numPatchesX),
      mNumPatchesY(numPatchesY),
      mNumPatchesF(numPatchesF),
      mNumArbors(numArbors),
      mCompressedFlag(compressedFlag),
      mReadOnly(readOnlyFlag),
      mVerifyWrites(verifyWrites) {
   initializeSharedWeightsIO();
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
   if (isRoot()) { mSharedWeightsIO->write(weightData, timestamp); }
   setIndex(mIndex + 1);
}

void SharedWeightsFile::truncate(int index) {
   FatalIf(
         mReadOnly,
         "SharedWeightsFile \"%s\" is read-only and cannot be truncated.\n",
         mPath.c_str());
   if (isRoot()) {
      int curFrameNumber = mSharedWeightsIO->getFrameNumber();
      int lastFrameNumber = mSharedWeightsIO->getNumFrames();
      if (index >= lastFrameNumber) {
         WarnLog().printf(
               "Attempt to truncate \"%s\" to index %d, but file's max index is only %d\n",
               mPath.c_str(), index, lastFrameNumber);
         return;
      }
      int newFrameNumber = curFrameNumber > index ? index : curFrameNumber;
      long eofPosition = mSharedWeightsIO->calcFilePositionFromFrameNumber(index);
      mSharedWeightsIO = std::unique_ptr<SharedWeightsIO>(); // closes existing file
      mFileManager->truncate(mPath, eofPosition);
      initializeSharedWeightsIO(); // reopens existing file with same mode.
      mSharedWeightsIO->setFrameNumber(newFrameNumber);
   }
}

void SharedWeightsFile::setIndex(int index) {
   mIndex = index;
   if (!isRoot()) { return; }
   int frameNumber = index;
   if (mReadOnly) {
      frameNumber = index % mSharedWeightsIO->getNumFrames();
   }
   mSharedWeightsIO->setFrameNumber(frameNumber);
}

void SharedWeightsFile::initializeSharedWeightsIO() {
   auto fileStream = FileStreamBuilder(
         mFileManager, mPath, false /*not text*/, mReadOnly, mVerifyWrites).get();

   mSharedWeightsIO = std::unique_ptr<SharedWeightsIO>(new SharedWeightsIO(
         fileStream,
         mPatchSizeX, mPatchSizeY, mPatchSizeF,
         mNumPatchesX, mNumPatchesY, mNumPatchesF,
         mNumArbors, mCompressedFlag));
}

void SharedWeightsFile::readInternal(WeightData &weightData, double &timestamp) {
   int status = PV_SUCCESS;
   if (isRoot()) {
      mSharedWeightsIO->read(weightData, timestamp);  
   }
   int numElements = getPatchSizeOverall() * getNumPatchesOverall();
   int rootProc = mFileManager->getRootProcessRank();
   auto mpiComm = mFileManager->getMPIBlock()->getComm();
   for (int a = 0; a < getNumArbors(); ++a) {
      float *weightsData = weightData.getData(a);
      MPI_Bcast(weightsData, numElements, MPI_FLOAT, rootProc, mpiComm);
   }
   setIndex(mIndex + 1);
}

} // namespace PV
