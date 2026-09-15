#include "LocalPatchWeightsFile.hpp"

#include "io/FileStreamBuilder.hpp"
#include "structures/Patch.hpp"

#include <algorithm> // std::copy()
#include <sys/stat.h>   // stat()
#include <sys/unistd.h> // sync()

namespace PV {

LocalPatchWeightsFile::LocalPatchWeightsFile(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &path,
      std::shared_ptr<WeightData> weightData,
      PVLayerLoc const *preLayerLoc,
      PVLayerLoc const *postLayerLoc,
      bool fileExtendedFlag,
      bool compressedFlag,
      bool readOnlyFlag,
      bool clobberFlag,
      bool verifyWrites)
      : WeightsFile(weightData),
        mFileManager(fileManager),
        mPath(path),
        mPatchSizeX(weightData->getPatchSizeX()),
        mPatchSizeY(weightData->getPatchSizeY()),
        mPatchSizeF(weightData->getPatchSizeF()),
        mPreLayerLoc(*preLayerLoc),
        mPostLayerLoc(*postLayerLoc),
        mNumArbors(weightData->getNumArbors()),
        mFileExtendedFlag(fileExtendedFlag),
        mCompressedFlag(compressedFlag),
        mReadOnlyFlag(readOnlyFlag),
        mVerifyWrites(verifyWrites) {
   initializeCheckpointerDataInterface();
   initializeWeightsIO(clobberFlag);
}

LocalPatchWeightsFile::~LocalPatchWeightsFile() {}

void LocalPatchWeightsFile::read() {
   double dummyTimestamp;
   readInternal(dummyTimestamp);
}

void LocalPatchWeightsFile::read(double &timestamp) {
   readInternal(timestamp);
   auto mpiComm = mFileManager->getMPIBlock()->getComm();
   MPI_Bcast(&timestamp, 1, MPI_DOUBLE, mFileManager->getRootProcessRank(), mpiComm);
}

void LocalPatchWeightsFile::write(double timestamp) {
   float extremeValues[2]; // extremeValues[0] is the min; extremeValues[1] is the max.
   mLocalPatchWeightsIO->calcExtremeWeights(
         *mWeightData,
         getNxRestrictedPre(),
         getNyRestrictedPre(),
         getNxRestrictedPost(),
         getNyRestrictedPost(),
         extremeValues[0],
         extremeValues[1]);
   int root         = mFileManager->getRootProcessRank();
   auto mpiBlock    = mFileManager->getMPIBlock();
   void *sendbuf    = isRoot() ? MPI_IN_PLACE : extremeValues;
   extremeValues[1] = -extremeValues[1]; // Use the same MPI_Reduce call to work on both min and max
   MPI_Reduce(sendbuf, extremeValues, 2, MPI_FLOAT, MPI_MIN, root, mpiBlock->getComm());
   extremeValues[1] = -extremeValues[1];
   long numValuesL  = mWeightData->getNumValuesPerArbor();
   int numValues    = static_cast<int>(numValuesL);
   FatalIf(
         static_cast<long>(numValues) != numValuesL,
         "Weights file \"%s\" must send/receive %ld values over MPI, which is too large.\n",
         mPath.c_str(), numValuesL);
   if (isRoot()) {
      BufferUtils::WeightHeader header =
            createHeader(timestamp, extremeValues[0], extremeValues[1]);
      mLocalPatchWeightsIO->writeHeader(header);
      WeightData tempWeightData(
            mPath,
            mWeightData->getNumArbors(),
            mWeightData->getPatchSizeX(),
            mWeightData->getPatchSizeY(),
            mWeightData->getPatchSizeF(),
            mWeightData->getNumDataPatchesX(),
            mWeightData->getNumDataPatchesY(),
            mWeightData->getNumDataPatchesF());
      for (int rank = 0; rank < mpiBlock->getSize(); ++rank) {
         if (rank == mpiBlock->getRank()) {
            continue;
         } // Leave local slice until the end
         for (int a = 0; a < tempWeightData.getNumArbors(); ++a) {
            float *arbor = tempWeightData.getData(a);
            int tag      = 136;
            MPI_Recv(
                  arbor, numValues, MPI_FLOAT, rank, tag, mpiBlock->getComm(), MPI_STATUS_IGNORE);
         }
         int xStartRestricted = mpiBlock->calcColumnFromRank(rank) * getNxRestrictedPre();
         int yStartRestricted = mpiBlock->calcRowFromRank(rank) * getNyRestrictedPre();
         mLocalPatchWeightsIO->writeRegion(
               tempWeightData,
               header,
               getNxRestrictedPre(),
               getNyRestrictedPre(),
               getNxRestrictedPost(),
               getNyRestrictedPost(),
               xStartRestricted,
               yStartRestricted,
               0 /*regionFStartRestricted*/,
               0 /*arborIndexStart*/);
      }
      // Now do local slice
      int xStartRestricted = mpiBlock->getColumnIndex() * getNxRestrictedPre();
      int yStartRestricted = mpiBlock->getRowIndex() * getNyRestrictedPre();
      mLocalPatchWeightsIO->writeRegion(
            *mWeightData,
            header,
            getNxRestrictedPre(),
            getNyRestrictedPre(),
            getNxRestrictedPost(),
            getNyRestrictedPost(),
            xStartRestricted,
            yStartRestricted,
            0 /*regionFStartRestricted*/,
            0 /*arborIndexStart*/);
   } // if (isRoot())
   else {
      for (int a = 0; a < getNumArbors(); ++a) {
         float const *arbor = mWeightData->getData(a);
         int tag            = 136;
         MPI_Send(arbor, numValues, MPI_FLOAT, root, tag, mpiBlock->getComm());
      }
   }
   if (isRoot()) {
      mLocalPatchWeightsIO->finishWrite();
   }
   MPI_Barrier(mpiBlock->getGlobalComm());
   pvAssert(mLocalPatchWeightsIO != nullptr);
   if (mSeesElemZeroFlag) {
      mLocalPatchWeightsIO->close();
      mLocalPatchWeightsIO->open();
   }
   setIndex(getIndex() + 1);
   InfoLog().printf(
         "Writing to \"%s\" finished; file has %d frames\n",
         mPath.c_str(), mLocalPatchWeightsIO->getNumFrames());
}

void LocalPatchWeightsFile::truncate(int index) {
   FatalIf(
         mReadOnlyFlag,
         "LocalPatchWeightsFile \"%s\" is read-only and cannot be truncated.\n",
         mPath.c_str());
   pvAssert(mLocalPatchWeightsIO != nullptr and mSharedWeightsIO == nullptr);
   if (isRoot()) {
      int curFrameNumber  = mLocalPatchWeightsIO->getFrameNumber();
      int lastFrameNumber = mLocalPatchWeightsIO->getNumFrames();
      if (index >= lastFrameNumber) {
         WarnLog().printf(
               "Attempt to truncate \"%s\" to index %d, but file's max index is only %d\n",
               mPath.c_str(),
               index,
               lastFrameNumber);
         return;
      }
      int newFrameNumber = curFrameNumber > index ? index : curFrameNumber;
      long eofPosition   = mLocalPatchWeightsIO->calcFilePositionFromFrameNumber(index);
      mLocalPatchWeightsIO->close();
      mFileManager->truncate(mPath, eofPosition);
      mLocalPatchWeightsIO->open();
      int newIndex = index < getIndex() ? index : getIndex();
      setIndex(newIndex);
   }
}

void LocalPatchWeightsFile::setIndex(int index) {
   if (!isRoot()) {
      return;
   }
   int frameNumber = index;
   if (mLocalPatchWeightsIO != nullptr) {
      pvAssert(mSharedWeightsIO == nullptr);
      mLocalPatchWeightsIO->setFrameNumber(index);
      frameNumber = mLocalPatchWeightsIO->getFrameNumber();
      mFileStreamReadPos = mLocalPatchWeightsIO->getFileStream()->getInPos();
      if (!mReadOnlyFlag) {
         mFileStreamWritePos = mLocalPatchWeightsIO->getFileStream()->getOutPos();
      }
      else {
         mFileStreamWritePos = mFileStreamReadPos;
      }
   }
   else {
      pvAssert(mSharedWeightsIO != nullptr);
      pvAssert(mReadOnlyFlag);
      mSharedWeightsIO->setFrameNumber(index);
      frameNumber = mSharedWeightsIO->getFrameNumber();
      mFileStreamReadPos = mSharedWeightsIO->getFileStream()->getInPos();
      mFileStreamWritePos = mFileStreamReadPos;
   }
   WeightsFile::setIndex(frameNumber);
}

Response::Status LocalPatchWeightsFile::registerData(
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

Response::Status LocalPatchWeightsFile::processCheckpointRead(double simTime) {
   auto status = CheckpointerDataInterface::processCheckpointRead(simTime);
   if (!Response::completed(status)) {
      return status;
   }
   long pos  = mReadOnlyFlag ? mFileStreamReadPos : mFileStreamWritePos;
   int index = mLocalPatchWeightsIO->calcFrameNumberFromFilePosition(pos);
   setIndex(index);
   if (isRoot() and mLocalPatchWeightsIO->getFrameNumber() < mLocalPatchWeightsIO->getNumFrames()) {
      WarnLog() << "Truncating \"" << getPath() << "\" to "
                << mLocalPatchWeightsIO->getFrameNumber() << " frames.\n";
      truncate(getIndex());
   }
   return Response::SUCCESS;
}

void LocalPatchWeightsFile::convertSharedToNonshared(WeightData const &sharedWeightData) {
   // Each process has the shared weights in the sharedWeightData argument. Now we need to
   // loop over all the nonshared patches, and read in the correct kernel.
   int const numPatchesX        = getNxExtendedPre();
   int const numPatchesY        = getNyExtendedPre();
   int const numPatchesF        = getNfPre();
   long const numPatchesOverall = (long)numPatchesX * (long)numPatchesY * (long)numPatchesF;
   int const numKernelsX        = sharedWeightData.getNumDataPatchesX();
   int const numKernelsY        = sharedWeightData.getNumDataPatchesY();
   int const numKernelsF        = sharedWeightData.getNumDataPatchesF();
   long const patchSizeOverall  = getPatchSizeOverall();
   for (int a = 0; a < mNumArbors; ++a) {
      for (long k = 0; k < numPatchesOverall; ++k) {
         int kxExt = kxPos(k, numPatchesX, numPatchesY, numPatchesF);
         int xCell = (kxExt - mPreLayerLoc.halo.lt + mPreLayerLoc.kx0) % numKernelsX;
         xCell += (xCell < 0) ? numKernelsX : 0;

         int kyExt = kyPos(k, numPatchesX, numPatchesY, numPatchesF);
         int yCell = (kyExt - mPreLayerLoc.halo.dn + mPreLayerLoc.ky0) % numKernelsY;
         yCell += (yCell < 0) ? numKernelsY : 0;

         int kf = featureIndex(k, numPatchesX, numPatchesY, numPatchesF);

         float const *sharedWeightValues = sharedWeightData.getDataFromXYF(a, xCell, yCell, kf);
         float *targetWeightValues = mWeightData->getDataFromXYF(a, kxExt, kyExt, kf);
         std::copy(sharedWeightValues, &sharedWeightValues[patchSizeOverall], targetWeightValues);
      }
   }
}

int LocalPatchWeightsFile::initializeCheckpointerDataInterface() {
   return CheckpointerDataInterface::initialize();
}

void LocalPatchWeightsFile::initializeWeightsIO(bool clobberFlag) {
   if (mReadOnlyFlag) {
      // The file must exist, and it must be a weights PVP file, but it could be
      // either SharedWeights or LocalPatchWeights.
      BufferUtils::WeightHeader weightHeader;
      int weightHeaderSize = static_cast<int>(sizeof(weightHeader));
      if (isRoot()) {
         bool fileExists = mFileManager->queryFileExists(mPath);
         FatalIf(
               !fileExists,
               "Read-only flag was set but file \"%s\" does not exist.\n",
               mPath.c_str());
         auto fileStream =
               FileStreamBuilder(
                     mFileManager,
                     mPath,
                     false /*not text*/,
                     true /*readOnlyFlag*/,
                     false /*clobberFlag*/,
                     false /*verifyWritesFlag*/).get();
         fileStream->read(&weightHeader, static_cast<long>(weightHeaderSize));
      }
      MPI_Bcast(
            &weightHeader,
            weightHeaderSize,
            MPI_BYTE,
            mFileManager->getRootProcessRank(),
            mFileManager->getMPIBlock()->getComm());
      switch(weightHeader.baseHeader.fileType) {
         case PVP_WGT_FILE_TYPE:
            initializeLocalPatchWeightsIO(clobberFlag);
            break;
         case PVP_KERNEL_FILE_TYPE:
            initializeSharedWeightsIO(clobberFlag, weightHeader);
            break;
         default:
            Fatal().printf("File \"%s\" is not a weights PVP file.\n");
            break;
      }
   }
   else {
      initializeLocalPatchWeightsIO(clobberFlag);
   }
}

void LocalPatchWeightsFile::initializeLocalPatchWeightsIO(bool clobberFlag) {
   auto mpiBlock             = mFileManager->getMPIBlock();
   int nxRestrictedPreBlock  = getNxRestrictedPre() * mpiBlock->getNumColumns();
   int nyRestrictedPreBlock  = getNyRestrictedPre() * mpiBlock->getNumRows();
   int nxRestrictedPostBlock, nyRestrictedPostBlock;
   if (mPostLayerLoc.bcast) {
      nxRestrictedPostBlock = 1;
      nyRestrictedPostBlock = 1;
   }
   else {
      nxRestrictedPostBlock = getNxRestrictedPost() * mpiBlock->getNumColumns();
      nyRestrictedPostBlock = getNyRestrictedPost() * mpiBlock->getNumRows();
   }

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
         int elem = mpiBlock->getStartBatch() / mpiBlock->getBatchDimension();

         std::string elem0Dir =
               FileManager::createBlockDirNameFromColRowElem(baseDirectory, col, row, 0);
         std::string elem0Path = elem0Dir + mPath;
         struct stat statbuf;
         int status = ::stat(elem0Path.c_str(), &statbuf);
         if (status == 0) {
            InfoLog().printf("<LocalPatchWeightsFile.cpp:%d> Setting SeesElemZeroFlag to TRUE\n", __LINE__);
            mSeesElemZeroFlag = true;
         }
         else {
            InfoLog().printf(
                  "<LocalPatchWeightsFile.cpp:%d> Setting SeesElemZeroFlag to FALSE (error %d: %s)\n",
                  __LINE__, errno, std::strerror(errno));
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
            if ((statbuf.st_mode & S_IFREG) != S_IFREG) {
               ErrorLog().printf("File \"%s\" exists but is not a regular file.\n", elem0Dir.c_str());
               InfoLog().printf("<LocalPatchWeightsFile.cpp:%d> Setting SeesElemZeroFlag to FALSE\n", __LINE__);
               mSeesElemZeroFlag = false;
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
         mSeesElemZeroFlag = false;
      }
      int seesElemZeroInt = mSeesElemZeroFlag ? 1 : 0;
      MPI_Bcast(&seesElemZeroInt, 1 /*count*/, MPI_INT, 0 /*root*/, mpiBlock->getComm());
      mSeesElemZeroFlag = (seesElemZeroInt != 0);
   } // mpiBlock->getStartBatch() != 0
   InfoLog().printf("SeesElemZeroFlag = %s\n", mSeesElemZeroFlag ? "TRUE" : "FALSE");
   InfoLog().printf(
         "Opening LocalPatchWeightsIO with fileStream = %s\n",
         fileStream ? fileStream->getFileName().c_str() : "(null)");
   mLocalPatchWeightsIO = std::unique_ptr<LocalPatchWeightsIO>(new LocalPatchWeightsIO(
         fileStream,
         mPatchSizeX,
         mPatchSizeY,
         mPatchSizeF,
         nxRestrictedPreBlock,
         nyRestrictedPreBlock,
         getNfPre(),
         nxRestrictedPostBlock,
         nyRestrictedPostBlock,
         mNumArbors,
         mFileExtendedFlag,
         mCompressedFlag,
         !mSeesElemZeroFlag));
   // If a process with nonzero batch index sees the file with batch index == 0 and the ReadOnlyFlag
   // is false, we never write, and we open the file anew on each read since batch-zero file might
   // have changed the contents.
}

void LocalPatchWeightsFile::initializeSharedWeightsIO(
      bool clobberFlag, BufferUtils::WeightHeader weightHeader) {
   auto fileStream =
         FileStreamBuilder(
               mFileManager, mPath, false /*not text*/, mReadOnlyFlag, clobberFlag, mVerifyWrites)
               .get();
   mSharedWeightsIO = std::unique_ptr<SharedWeightsIO>(new SharedWeightsIO(
            fileStream,
            mPatchSizeX,
            mPatchSizeY,
            mPatchSizeF,
            weightHeader.baseHeader.nx,
            weightHeader.baseHeader.ny,
            getNfPre(),
            mNumArbors,
            mCompressedFlag));
}

void LocalPatchWeightsFile::readInternal(double &timestamp) {
   pvAssert(mLocalPatchWeightsIO != nullptr xor mSharedWeightsIO != nullptr);
   if (mLocalPatchWeightsIO != nullptr) {
      readLocalPatchWeights(timestamp);
   }
   else if (mSharedWeightsIO != nullptr) {
      readSharedWeights(timestamp);
   }
}

void LocalPatchWeightsFile::readLocalPatchWeights(double &timestamp) {
   long numValuesL = mWeightData->getNumValuesPerArbor();
   int numValues   = static_cast<int>(numValuesL);
   FatalIf(
         static_cast<long>(numValues) != numValuesL,
         "Weights file \"%s\" must send/receive %ld values over MPI, which is too large.\n",
         mPath.c_str(), numValuesL);
   auto mpiBlock = mFileManager->getMPIBlock();
   if (isRoot()) {
      BufferUtils::WeightHeader header = mLocalPatchWeightsIO->readHeader();
      timestamp                        = header.baseHeader.timestamp;
      // Need to check that header and weightData dimensions are compatible.
      int fileNxRestricted  = header.baseHeader.nx;
      int fileNyRestricted  = header.baseHeader.ny;
      int fileNxExtended    = header.baseHeader.nxExtended;
      int fileNyExtended    = header.baseHeader.nyExtended;
      int localNxRestricted = fileNxRestricted / mpiBlock->getNumColumns();
      int localNyRestricted = fileNyRestricted / mpiBlock->getNumRows();
      for (int rank = 0; rank < mpiBlock->getSize(); ++rank) {
         if (rank == mpiBlock->getRank()) {
            continue;
         } // Leave local slice until the end
         int xStartRestricted = mpiBlock->calcColumnFromRank(rank) * localNxRestricted;
         int yStartRestricted = mpiBlock->calcRowFromRank(rank) * localNyRestricted;
         mLocalPatchWeightsIO->readRegion(
               *mWeightData,
               header,
               localNxRestricted,
               localNyRestricted,
               getNxRestrictedPost(),
               getNyRestrictedPost(),
               xStartRestricted,
               yStartRestricted,
               0 /*regionFStartRestricted*/,
               0 /*arborIndexStart*/);
         for (int a = 0; a < mWeightData->getNumArbors(); ++a) {
            float *arbor = mWeightData->getData(a);
            int tag      = 134;
            MPI_Send(arbor, numValues, MPI_FLOAT, rank, tag, mpiBlock->getComm());
         }
      }
      // Now do local slice
      int xStartRestricted = mpiBlock->getColumnIndex() * localNxRestricted;
      int yStartRestricted = mpiBlock->getRowIndex() * localNyRestricted;
      mLocalPatchWeightsIO->readRegion(
            *mWeightData,
            header,
            localNxRestricted,
            localNyRestricted,
            getNxRestrictedPost(),
            getNyRestrictedPost(),
            xStartRestricted,
            yStartRestricted,
            0 /*regionFStartRestricted*/,
            0 /*arborIndexStart*/);
      mLocalPatchWeightsIO->setFrameNumber(mLocalPatchWeightsIO->getFrameNumber() + 1);
   }
   else {
      for (int a = 0; a < mWeightData->getNumArbors(); ++a) {
         float *arbor = mWeightData->getData(a);
         int root     = mFileManager->getRootProcessRank();
         int tag      = 134;
         MPI_Recv(arbor, numValues, MPI_FLOAT, root, tag, mpiBlock->getComm(), MPI_STATUS_IGNORE);
      }
   }
   setIndex(getIndex() + 1);
}

void LocalPatchWeightsFile::readSharedWeights(double &timestamp) {
   WeightData sharedWeightData(
         mPath,
         mNumArbors, mPatchSizeX, mPatchSizeY, mPatchSizeF,
         mSharedWeightsIO->getNumPatchesX(),
         mSharedWeightsIO->getNumPatchesY(),
         mSharedWeightsIO->getNumPatchesF());
   if (isRoot()) {
      mSharedWeightsIO->read(sharedWeightData, timestamp);
   }

   long numElementsL = sharedWeightData.getNumValuesPerArbor();
   int numElements   = static_cast<int>(numElementsL);
   FatalIf(
         static_cast<long>(numElements) != numElementsL,
         "Weights file \"%s\" must broadcast %ld values over MPI, which is too large.\n",
         mPath.c_str(), numElementsL);

   int rootProc = mFileManager->getRootProcessRank();
   auto mpiComm = mFileManager->getMPIBlock()->getComm();
   for (int a = 0; a < mNumArbors; ++a) {
      float *weightValues = sharedWeightData.getData(a);
      MPI_Bcast(weightValues, numElements, MPI_FLOAT, rootProc, mpiComm);
   }
   convertSharedToNonshared(sharedWeightData);
   setIndex(getIndex() + 1);
}

BufferUtils::WeightHeader
LocalPatchWeightsFile::createHeader(double timestamp, float minWgt, float maxWgt) const {
   BufferUtils::WeightHeader weightHeader;
   auto mpiBlock               = mFileManager->getMPIBlock();
   int nxRestrictedPreGathered = getNxRestrictedPre() * mpiBlock->getNumColumns();
   int nyRestrictedPreGathered = getNyRestrictedPre() * mpiBlock->getNumRows();
   int nxExtendedPreGathered   = nxRestrictedPreGathered + 2 * mLocalPatchWeightsIO->getXMargin();
   int nyExtendedPreGathered   = nyRestrictedPreGathered + 2 * mLocalPatchWeightsIO->getYMargin();

   weightHeader.baseHeader.headerSize = NUM_WGT_PARAMS * static_cast<int>(sizeof(float));
   weightHeader.baseHeader.numParams  = NUM_WGT_PARAMS;
   weightHeader.baseHeader.fileType   = PVP_WGT_FILE_TYPE;
   weightHeader.baseHeader.nx         = nxRestrictedPreGathered;
   weightHeader.baseHeader.ny         = nyRestrictedPreGathered;
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
   weightHeader.baseHeader.nxExtended = nxExtendedPreGathered;
   weightHeader.baseHeader.nyExtended = nyExtendedPreGathered;
   weightHeader.baseHeader.kx0        = 0;
   weightHeader.baseHeader.ky0        = 0;
   weightHeader.baseHeader.nBatch     = 1;
   weightHeader.baseHeader.nBands     = getNumArbors();
   weightHeader.baseHeader.timestamp  = timestamp;

   weightHeader.nxp        = getPatchSizeX();
   weightHeader.nyp        = getPatchSizeY();
   weightHeader.nfp        = getPatchSizeF();
   weightHeader.minVal     = minWgt;
   weightHeader.maxVal     = maxWgt;
   int numPatches          = nxExtendedPreGathered * nyExtendedPreGathered * getNfPre();
   weightHeader.numPatches = numPatches;
   return weightHeader;
}

} // namespace PV
