#include "BroadcastPreWeightsFile.hpp"

#include "io/FileStreamBuilder.hpp"

namespace PV {

BroadcastPreWeightsFile::BroadcastPreWeightsFile(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &path,
      std::shared_ptr<WeightData> weightData,
      int nfPre,
      bool compressedFlag,
      bool readOnlyFlag,
      bool clobberFlag,
      bool verifyWrites)
      : WeightsFile(weightData),
        mFileManager(fileManager),
        mPath(path),
        mPatchSizePerProcX(weightData->getPatchSizeX()),
        mPatchSizePerProcY(weightData->getPatchSizeY()),
        mPatchSizeF(weightData->getPatchSizeF()),
        mNfPre(nfPre),
        mNumArbors(weightData->getNumArbors()),
        mCompressedFlag(compressedFlag),
        mReadOnly(readOnlyFlag),
        mVerifyWrites(verifyWrites) {
   initializeCheckpointerDataInterface();
   initializeBroadcastPreWeightsIO(clobberFlag);
}

BroadcastPreWeightsFile::~BroadcastPreWeightsFile() {}

void BroadcastPreWeightsFile::read() {
   double dummyTimestamp;
   readInternal(dummyTimestamp);
}

void BroadcastPreWeightsFile::read(double &timestamp) {
   readInternal(timestamp);
   auto mpiComm = mFileManager->getMPIBlock()->getComm();
   MPI_Bcast(&timestamp, 1, MPI_DOUBLE, mFileManager->getRootProcessRank(), mpiComm);
}

void BroadcastPreWeightsFile::write(double timestamp) {
   float extremeValues[2]; // extremeValues[0] is the min; extremeValues[1] is the max.
   mWeightData->calcExtremeWeights(extremeValues[0], extremeValues[1]);
   int root         = mFileManager->getRootProcessRank();
   auto mpiBlock    = mFileManager->getMPIBlock();
   void *sendbuf    = isRoot() ? MPI_IN_PLACE : extremeValues;
   extremeValues[1] = -extremeValues[1]; // Use the same MPI_Reduce call to work on both min and max
   MPI_Reduce(sendbuf, extremeValues, 2, MPI_FLOAT, MPI_MIN, root, mpiBlock->getComm());
   extremeValues[1] = -extremeValues[1];
   long numValues   = mWeightData->getNumValuesPerArbor();
   if (isRoot()) {
      mBroadcastPreWeightsIO->setHeaderTimestamp(timestamp);
      mBroadcastPreWeightsIO->setHeaderExtremeVals(extremeValues[0], extremeValues[1]);
      mBroadcastPreWeightsIO->writeHeader();
      WeightData tempWeightData(
            mWeightData->getNumArbors(),
            mWeightData->getPatchSizeX(),
            mWeightData->getPatchSizeY(),
            mWeightData->getPatchSizeF(),
            mWeightData->getNumDataPatchesX(),
            mWeightData->getNumDataPatchesY(),
            mWeightData->getNumDataPatchesF());
      int nxpLocal = getPatchSizePerProcX();
      int nypLocal = getPatchSizePerProcY();
      for (int rank = 0; rank < mpiBlock->getSize(); ++rank) {
         if (rank == mpiBlock->getRank()) {
            continue;
         } // Leave local slice until the end
         float *weightData = tempWeightData.getData(0 /*arbor*/);
         int tag           = 136;
         MPI_Recv(
               weightData, numValues, MPI_FLOAT, rank, tag, mpiBlock->getComm(), MPI_STATUS_IGNORE);
         int xStart = nxpLocal * mpiBlock->calcColumnFromRank(rank);
         int yStart = nypLocal * mpiBlock->calcRowFromRank(rank);
         mBroadcastPreWeightsIO->writeRegion(
               tempWeightData,
               xStart,
               yStart,
               0 /*fStart*/,
               0 /*fPreStart*/,
               0 /*arborIndexStart*/);
      }
      // Now do local slice
      int xStart = mpiBlock->getColumnIndex();
      int yStart = mpiBlock->getRowIndex();
      mBroadcastPreWeightsIO->writeRegion(
            *mWeightData,
            xStart,
            yStart,
            0 /*fStart*/,
            0 /*fPreStart*/,
            0 /*arborIndexStart*/);
      mBroadcastPreWeightsIO->finishWrite();
   }
   else {
      float const *weightData = mWeightData->getData(0 /*arbor*/);
      int tag                 = 136;
      MPI_Send(weightData, numValues, MPI_FLOAT, root, tag, mpiBlock->getComm());
   }
   setIndex(getIndex() + 1);
}

void BroadcastPreWeightsFile::truncate(int index) {
   FatalIf(
         mReadOnly,
         "BroadcastPreWeightsFile \"%s\" is read-only and cannot be truncated.\n",
         mPath.c_str());
   if (isRoot()) {
      int curFrameNumber  = mBroadcastPreWeightsIO->getFrameNumber();
      int lastFrameNumber = mBroadcastPreWeightsIO->getNumFrames();
      if (index >= lastFrameNumber) {
         WarnLog().printf(
               "Attempt to truncate \"%s\" to index %d, but file's max index is only %d\n",
               mPath.c_str(),
               index,
               lastFrameNumber);
         return;
      }
      int newFrameNumber = curFrameNumber > index ? index : curFrameNumber;
      long eofPosition   = mBroadcastPreWeightsIO->calcFilePositionFromFrameNumber(index);
      mBroadcastPreWeightsIO->close();
      mFileManager->truncate(mPath, eofPosition);
      mBroadcastPreWeightsIO->open();
      int newIndex = index < getIndex() ? index : getIndex();
      setIndex(newIndex);
   }
}

void BroadcastPreWeightsFile::setIndex(int index) {
   WeightsFile::setIndex(index);
   if (!isRoot()) {
      return;
   }
   int frameNumber = index;
   if (mReadOnly) {
      frameNumber = index % mBroadcastPreWeightsIO->getNumFrames();
   }
   mBroadcastPreWeightsIO->setFrameNumber(frameNumber);
   mFileStreamReadPos = mBroadcastPreWeightsIO->getFileStream()->getInPos();
   if (!mReadOnly) {
      mFileStreamWritePos = mBroadcastPreWeightsIO->getFileStream()->getOutPos();
   }
   else {
      mFileStreamWritePos = mFileStreamReadPos;
   }
}

Response::Status BroadcastPreWeightsFile::registerData(
      std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) {
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

Response::Status BroadcastPreWeightsFile::processCheckpointRead(double simTime) {
   auto status = CheckpointerDataInterface::processCheckpointRead(simTime);
   if (!Response::completed(status)) {
      return status;
   }
   long pos  = mReadOnly ? mFileStreamReadPos : mFileStreamWritePos;
   int index = mBroadcastPreWeightsIO->calcFrameNumberFromFilePosition(pos);
   setIndex(index);
   if (isRoot() and mBroadcastPreWeightsIO->getFrameNumber() < mBroadcastPreWeightsIO->getNumFrames()) {
      WarnLog() << "Truncating \"" << getPath() << "\" to "
                << mBroadcastPreWeightsIO->getFrameNumber() << " frames.\n";
      truncate(getIndex());
   }
   return Response::SUCCESS;
}

int BroadcastPreWeightsFile::initializeCheckpointerDataInterface() {
   return CheckpointerDataInterface::initialize();
}

void BroadcastPreWeightsFile::initializeBroadcastPreWeightsIO(bool clobberFlag) {
   auto fileStream =
         FileStreamBuilder(
               mFileManager, mPath, false /*not text*/, mReadOnly, clobberFlag, mVerifyWrites)
               .get();
   auto mpiBlock             = mFileManager->getMPIBlock();

   mBroadcastPreWeightsIO = std::unique_ptr<BroadcastPreWeightsIO>(new BroadcastPreWeightsIO(
         fileStream,
         mPatchSizePerProcX * mpiBlock->getNumColumns(),
         mPatchSizePerProcY * mpiBlock->getNumRows(),
         mPatchSizeF,
         mNfPre,
         mNumArbors,
         mCompressedFlag));
}

void BroadcastPreWeightsFile::readInternal(double &timestamp) {
   long numValues = mWeightData->getNumValuesPerArbor();
   auto mpiBlock  = mFileManager->getMPIBlock();
   if (isRoot()) {
      mBroadcastPreWeightsIO->readHeader();
      timestamp = mBroadcastPreWeightsIO->getHeaderTimestamp();

      int nxpLocal = getPatchSizePerProcX();
      int nypLocal = getPatchSizePerProcY();
      for (int rank = 0; rank < mpiBlock->getSize(); ++rank) {
         if (rank == mpiBlock->getRank()) {
            continue;
         } // Leave local slice until the end
         int xStart = nxpLocal * mpiBlock->calcColumnFromRank(rank);
         int yStart = nypLocal * mpiBlock->calcRowFromRank(rank);
         mBroadcastPreWeightsIO->readRegion(
               *mWeightData,
               xStart,
               yStart,
               0 /*fStart*/,
               0 /*fPreStart*/,
               0 /*arborIndexStart*/);
         float *weightData = mWeightData->getData(0 /*arbor*/);
         int tag           = 134;
         MPI_Send(weightData, numValues, MPI_FLOAT, rank, tag, mpiBlock->getComm());
      }
      // Now do local slice
      int xStart = nxpLocal * mpiBlock->getColumnIndex();
      int yStart = nypLocal * mpiBlock->getRowIndex();
      mBroadcastPreWeightsIO->readRegion(
            *mWeightData,
            xStart,
            yStart,
            0 /*fStart*/,
            0 /*fPreStart*/,
            0 /*arborIndexStart*/);
      mBroadcastPreWeightsIO->setFrameNumber(mBroadcastPreWeightsIO->getFrameNumber() + 1);
   }
   else {
      float *weightData = mWeightData->getData(0 /*arbor*/);
      int root     = mFileManager->getRootProcessRank();
      int tag      = 134;
      MPI_Recv(weightData, numValues, MPI_FLOAT, root, tag, mpiBlock->getComm(), MPI_STATUS_IGNORE);
   }
   setIndex(getIndex() + 1);
}

BufferUtils::WeightHeader
BroadcastPreWeightsFile::createHeader(double timestamp, float minWgt, float maxWgt) const {
   BufferUtils::WeightHeader weightHeader;
   auto mpiBlock               = mFileManager->getMPIBlock();

   weightHeader.baseHeader.headerSize = NUM_WGT_PARAMS * static_cast<int>(sizeof(float));
   weightHeader.baseHeader.numParams  = NUM_WGT_PARAMS;
   weightHeader.baseHeader.fileType   = PVP_WGT_FILE_TYPE;
   weightHeader.baseHeader.nx         = 1;
   weightHeader.baseHeader.ny         = 1;
   weightHeader.baseHeader.nf         = getNfPre();
   weightHeader.baseHeader.numRecords = getNumArbors();
   weightHeader.baseHeader.recordSize = 0;
   if (getCompressedFlag()) {
      weightHeader.baseHeader.dataSize = static_cast<int>(sizeof(uint8_t));
      weightHeader.baseHeader.dataType = BufferUtils::BYTE;
   }
   else {
      weightHeader.baseHeader.dataSize = static_cast<int>(sizeof(float));
      weightHeader.baseHeader.dataType = BufferUtils::FLOAT;
   }
   weightHeader.baseHeader.nxProcs    = 1;
   weightHeader.baseHeader.nyProcs    = 1;
   weightHeader.baseHeader.nxExtended = 1;
   weightHeader.baseHeader.nyExtended = 1;
   weightHeader.baseHeader.kx0        = 0;
   weightHeader.baseHeader.ky0        = 0;
   weightHeader.baseHeader.nBatch     = 1;
   weightHeader.baseHeader.nBands     = getNumArbors();
   weightHeader.baseHeader.timestamp  = timestamp;

   weightHeader.nxp        = getPatchSizePerProcX() * mpiBlock->getNumColumns();
   weightHeader.nyp        = getPatchSizePerProcY() * mpiBlock->getNumRows();
   weightHeader.nfp        = getPatchSizeF();
   weightHeader.minVal     = minWgt;
   weightHeader.maxVal     = maxWgt;
   weightHeader.numPatches = getNfPre();
   return weightHeader;
}

} // namespace PV
