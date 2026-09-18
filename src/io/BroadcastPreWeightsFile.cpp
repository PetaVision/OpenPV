#include "BroadcastPreWeightsFile.hpp"

#include "io/FileStreamBuilder.hpp"

namespace PV {

BroadcastPreWeightsFile::BroadcastPreWeightsFile(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &path,
      std::shared_ptr<WeightData> weightData,
      int nfPre,
      bool postIsBroadcastFlag,
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
        mPostIsBroadcastFlag(postIsBroadcastFlag),
        mCompressedFlag(compressedFlag),
        mReadOnlyFlag(readOnlyFlag),
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
   if (getPostIsBroadcastFlag()) {
      writePostIsBroadcast(timestamp);
   }
   else {
      writePostIsNotBroadcast(timestamp);
   }
}

void BroadcastPreWeightsFile::writePostIsBroadcast(double timestamp) {
   float minValue, maxValue;
   mWeightData->calcExtremeWeights(minValue, maxValue);
   if (isRoot()) {
      mBroadcastPreWeightsIO->setHeaderTimestamp(timestamp);
      mBroadcastPreWeightsIO->setHeaderExtremeVals(minValue, maxValue);
      mBroadcastPreWeightsIO->writeHeader();
      mBroadcastPreWeightsIO->writeRegion(
            *mWeightData,
            0 /*xStart*/,
            0 /*yStart*/,
            0 /*fStart*/,
            0 /*fPreStart*/,
            0 /*arborIndexStart*/);
      mBroadcastPreWeightsIO->finishWrite();
   }
   setIndex(getIndex() + 1);
}

void BroadcastPreWeightsFile::writePostIsNotBroadcast(double timestamp) {
   float extremeValues[2]; // extremeValues[0] is the min; extremeValues[1] is the max.
   mWeightData->calcExtremeWeights(extremeValues[0], extremeValues[1]);
   int root         = mFileManager->getRootProcessRank();
   auto mpiBlock    = mFileManager->getMPIBlock();
   void *sendbuf    = isRoot() ? MPI_IN_PLACE : extremeValues;
   extremeValues[1] = -extremeValues[1]; // Use the same MPI_Reduce call to work on both min and max
   MPI_Reduce(sendbuf, extremeValues, 2, MPI_FLOAT, MPI_MIN, root, mpiBlock->getComm());
   extremeValues[1] = -extremeValues[1];
   long numValues   = mWeightData->getNumValuesPerArbor();
   int numValuesMPI = static_cast<int>(numValues); // MPI_Send/Recv take args of type int.
   FatalIf(
         static_cast<long>(numValuesMPI) != numValues,
         "Writing \"%s\" requires MPI_Send/Recv of %ld values, which is larger than INT_MAX=%d\n",
         mPath.c_str(), numValues, INT_MAX);
   if (isRoot()) {
      mBroadcastPreWeightsIO->setHeaderTimestamp(timestamp);
      mBroadcastPreWeightsIO->setHeaderExtremeVals(extremeValues[0], extremeValues[1]);
      mBroadcastPreWeightsIO->writeHeader();
      WeightData tempWeightData(
            mPath,
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
               weightData,
               numValuesMPI,
               MPI_FLOAT,
               rank,
               tag,
               mpiBlock->getComm(),
               MPI_STATUS_IGNORE);
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
      MPI_Send(weightData, numValuesMPI, MPI_FLOAT, root, tag, mpiBlock->getComm());
   }
   MPI_Barrier(mpiBlock->getGlobalComm());
   setIndex(getIndex() + 1);
}

void BroadcastPreWeightsFile::truncate(int index) {
   FatalIf(
         mReadOnlyFlag,
         "BroadcastPreWeightsFile \"%s\" is read-only and cannot be truncated.\n",
         mPath.c_str());
   if (isRoot()) {
      if (!mSeesElemZeroFlag) {
         int lastFrameNumber = mBroadcastPreWeightsIO->getNumFrames();
         if (index >= lastFrameNumber) {
            WarnLog().printf(
                  "Attempt to truncate \"%s\" to index %d, but file's max index is only %d\n",
                  mPath.c_str(),
                  index,
                  lastFrameNumber);
            return;
         }
      }
   }
   long newEof = mBroadcastPreWeightsIO->calcFilePositionFromFrameNumber(index);
   mBroadcastPreWeightsIO->close();
   if (!mSeesElemZeroFlag) {
      mFileManager->truncate(mPath, newEof);
   }
   MPI_Barrier(mFileManager->getMPIBlock()->getGlobalComm());
   mBroadcastPreWeightsIO->open();
   int newIndex = index < getIndex() ? index : getIndex();
   setIndex(newIndex);
}

void BroadcastPreWeightsFile::setIndex(int index) {
   WeightsFile::setIndex(index);
   if (!isRoot()) {
      return;
   }
   int frameNumber = index;
   if (mReadOnlyFlag) {
      frameNumber = index % mBroadcastPreWeightsIO->getNumFrames();
   }
   mBroadcastPreWeightsIO->setFrameNumber(frameNumber);
   mFileStreamReadPos = mBroadcastPreWeightsIO->getFileStream()->getInPos();
   if (!mReadOnlyFlag) {
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
   long pos  = mReadOnlyFlag ? mFileStreamReadPos : mFileStreamWritePos;
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
   auto mpiBlock             = mFileManager->getMPIBlock();
   std::shared_ptr<FileStream> fileStream = nullptr;
   if (mpiBlock->getStartBatch() == 0) {
      fileStream = FileStreamBuilder(
            mFileManager, mPath, false /*isTextFlag*/, mReadOnlyFlag, clobberFlag, mVerifyWrites)
            .get();
      mSeesElemZeroFlag = false;
      sync();
   }
   // Make sure the batch-zero process has created the file before other processes check for
   // its existence.
   MPI_Barrier(mpiBlock->getGlobalComm());
   if (mpiBlock->getStartBatch() != 0) {
      if (isRoot()) {
         std::string effectivePath = mFileManager->convertToEffectivePath(mPath);
         std::string const &baseDirectory = mFileManager->getBaseDirectory();
         int col  = mpiBlock->getStartColumn() / mpiBlock->getNumColumns();
         int row  = mpiBlock->getStartRow() / mpiBlock->getNumRows();

         std::string elem0Dir =
               FileManager::createBlockDirNameFromColRowElem(baseDirectory, col, row, 0);
         std::string elem0Path = elem0Dir + mPath;
         struct stat statbuf;
         int status = ::stat(elem0Path.c_str(), &statbuf);
         if (status == 0) {
            if ((statbuf.st_mode & S_IFREG) == S_IFREG) {
               mSeesElemZeroFlag = true;
            }
            else {
               ErrorLog().printf("File \"%s\" exists but is not a regular file.\n", elem0Dir.c_str());
               mSeesElemZeroFlag = false;
            }
         }
         else {
            mSeesElemZeroFlag = false;
            if (errno == ENOENT) {
               errno = 0;
            }
            else {
               ErrorLog().printf(
                     "Unable to query existence of file \"%s\": error %d (%s).\n",
                     elem0Dir.c_str(), errno, std::strerror(errno));
            }
         }
         if (mSeesElemZeroFlag) {
            mElemZeroPath = elem0Path;
            InfoLog().printf(
                  "Using batch-zero file \"%s\" instead of \"%s\"\n",
                  elem0Path.c_str(), effectivePath.c_str());
         }
         if (mSeesElemZeroFlag) {
            if (isRoot()) {
               std::ios_base::openmode mode = std::ios_base::in | std::ios_base::binary;
               if (!mReadOnlyFlag) { mode |= std::ios_base::out; }
               fileStream = std::make_shared<FileStream>(
                     mElemZeroPath.c_str(), mode, mVerifyWrites);
            }
            else {
               fileStream = nullptr;
            }
         }
         else {
            fileStream = FileStreamBuilder(
               mFileManager, mPath, false /*isTextFlag*/, mReadOnlyFlag, clobberFlag, mVerifyWrites)
               .get();
         }
      } // if (isRoot())
      else {
         mSeesElemZeroFlag = false; // eliminate uninitialized variable warning
      }
      int seesElemZeroInt = mSeesElemZeroFlag ? 1 : 0;
      MPI_Bcast(&seesElemZeroInt, 1 /*count*/, MPI_INT, 0 /*root*/, mpiBlock->getComm());
      mSeesElemZeroFlag = (seesElemZeroInt != 0);
   }

   int ioPatchSizeX = mPatchSizePerProcX;
   int ioPatchSizeY = mPatchSizePerProcY;
   if (!getPostIsBroadcastFlag()) {
      auto mpiBlock = mFileManager->getMPIBlock();
      ioPatchSizeX *= mpiBlock->getNumColumns();
      ioPatchSizeY *= mpiBlock->getNumRows();
   }

   mBroadcastPreWeightsIO = std::unique_ptr<BroadcastPreWeightsIO>(new BroadcastPreWeightsIO(
         fileStream,
         ioPatchSizeX,
         ioPatchSizeY,
         mPatchSizeF,
         mNfPre,
         mNumArbors,
         mCompressedFlag,
         !mSeesElemZeroFlag));
}

void BroadcastPreWeightsFile::readInternal(double &timestamp) {
   if (getPostIsBroadcastFlag()) {
      readPostIsBroadcast(timestamp);
   }
   else {
      readPostIsNotBroadcast(timestamp);
   }
}

void BroadcastPreWeightsFile::readPostIsBroadcast(double &timestamp) {
   long numValuesPerArbor = mWeightData->getNumValuesPerArbor();
   int broadcastCount     = static_cast<int>(numValuesPerArbor);
   FatalIf(
         static_cast<long>(broadcastCount) != numValuesPerArbor,
         "Reading \"%s\" requires MPI broadcast of %ld values, which is larger than INT_MAX=%d\n",
         mPath.c_str(), numValuesPerArbor, INT_MAX);
   if (isRoot()) {
      mBroadcastPreWeightsIO->readHeader();
      timestamp = mBroadcastPreWeightsIO->getHeaderTimestamp();
      mBroadcastPreWeightsIO->readRegion(
            *mWeightData,
            0 /*xStart*/,
            0 /*yStart*/,
            0 /*fStart*/,
            0 /*fPreStart*/,
            0 /*arborIndexStart*/);
   }
   MPI_Bcast(
         mWeightData->getData(0 /*arbor*/),
         broadcastCount,
         MPI_FLOAT,
         mFileManager->getRootProcessRank(),
         mFileManager->getMPIBlock()->getComm());
   setIndex(getIndex() + 1);
}

void BroadcastPreWeightsFile::readPostIsNotBroadcast(double &timestamp) {
   long numValuesPerArbor = mWeightData->getNumValuesPerArbor();
   int sendCount          = static_cast<int>(numValuesPerArbor);
   FatalIf(
         static_cast<long>(sendCount) != numValuesPerArbor,
         "Reading \"%s\" requires MPI send/recv of %ld values, which is larger than INT_MAX=%d\n",
         mPath.c_str(), numValuesPerArbor, INT_MAX);
   auto mpiBlock          = mFileManager->getMPIBlock();
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
         MPI_Send(weightData, sendCount, MPI_FLOAT, rank, tag, mpiBlock->getComm());
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
      MPI_Recv(weightData, sendCount, MPI_FLOAT, root, tag, mpiBlock->getComm(), MPI_STATUS_IGNORE);
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
