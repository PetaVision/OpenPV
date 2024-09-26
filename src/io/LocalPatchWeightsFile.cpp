#include "LocalPatchWeightsFile.hpp"

#include "io/FileStreamBuilder.hpp"

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
        mNxRestrictedPre(preLayerLoc->nx),
        mNyRestrictedPre(preLayerLoc->ny),
        mNfPre(preLayerLoc->nf),
        mNxRestrictedPost(postLayerLoc->nx),
        mNyRestrictedPost(postLayerLoc->ny),
        mNumArbors(weightData->getNumArbors()),
        mFileExtendedFlag(fileExtendedFlag),
        mCompressedFlag(compressedFlag),
        mReadOnly(readOnlyFlag),
        mVerifyWrites(verifyWrites) {
   initializeCheckpointerDataInterface();
   initializeLocalPatchWeightsIO(clobberFlag);
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
         extremeValues[0],
         extremeValues[1]);
   int root         = mFileManager->getRootProcessRank();
   auto mpiBlock    = mFileManager->getMPIBlock();
   void *sendbuf    = isRoot() ? MPI_IN_PLACE : extremeValues;
   extremeValues[1] = -extremeValues[1]; // Use the same MPI_Reduce call to work on both min and max
   MPI_Reduce(sendbuf, extremeValues, 2, MPI_FLOAT, MPI_MIN, root, mpiBlock->getComm());
   extremeValues[1] = -extremeValues[1];
   long numValues   = mWeightData->getNumValuesPerArbor();
   if (isRoot()) {
      BufferUtils::WeightHeader header =
            createHeader(timestamp, extremeValues[0], extremeValues[1]);
      mLocalPatchWeightsIO->writeHeader(header);
      WeightData tempWeightData(
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
            xStartRestricted,
            yStartRestricted,
            0 /*regionFStartRestricted*/,
            0 /*arborIndexStart*/);
      mLocalPatchWeightsIO->finishWrite();
   }
   else {
      for (int a = 0; a < getNumArbors(); ++a) {
         float const *arbor = mWeightData->getData(a);
         int tag            = 136;
         MPI_Send(arbor, numValues, MPI_FLOAT, root, tag, mpiBlock->getComm());
      }
   }
   setIndex(getIndex() + 1);
}

void LocalPatchWeightsFile::truncate(int index) {
   FatalIf(
         mReadOnly,
         "LocalPatchWeightsFile \"%s\" is read-only and cannot be truncated.\n",
         mPath.c_str());
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
   WeightsFile::setIndex(index);
   if (!isRoot()) {
      return;
   }
   int frameNumber = index;
   if (mReadOnly) {
      frameNumber = index % mLocalPatchWeightsIO->getNumFrames();
   }
   mLocalPatchWeightsIO->setFrameNumber(frameNumber);
   mFileStreamReadPos = mLocalPatchWeightsIO->getFileStream()->getInPos();
   if (!mReadOnly) {
      mFileStreamWritePos = mLocalPatchWeightsIO->getFileStream()->getOutPos();
   }
   else {
      mFileStreamWritePos = mFileStreamReadPos;
   }
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
   long pos  = mReadOnly ? mFileStreamReadPos : mFileStreamWritePos;
   int index = mLocalPatchWeightsIO->calcFrameNumberFromFilePosition(pos);
   setIndex(index);
   if (isRoot() and mLocalPatchWeightsIO->getFrameNumber() < mLocalPatchWeightsIO->getNumFrames()) {
      WarnLog() << "Truncating \"" << getPath() << "\" to "
                << mLocalPatchWeightsIO->getFrameNumber() << " frames.\n";
      truncate(getIndex());
   }
   return Response::SUCCESS;
}

int LocalPatchWeightsFile::initializeCheckpointerDataInterface() {
   return CheckpointerDataInterface::initialize();
}

void LocalPatchWeightsFile::initializeLocalPatchWeightsIO(bool clobberFlag) {
   auto fileStream =
         FileStreamBuilder(
               mFileManager, mPath, false /*not text*/, mReadOnly, clobberFlag, mVerifyWrites)
               .get();
   auto mpiBlock             = mFileManager->getMPIBlock();
   int nxRestrictedPreBlock  = mNxRestrictedPre * mpiBlock->getNumColumns();
   int nyRestrictedPreBlock  = mNyRestrictedPre * mpiBlock->getNumRows();
   int nxRestrictedPostBlock = mNxRestrictedPost * mpiBlock->getNumColumns();
   int nyRestrictedPostBlock = mNyRestrictedPost * mpiBlock->getNumRows();

   mLocalPatchWeightsIO = std::unique_ptr<LocalPatchWeightsIO>(new LocalPatchWeightsIO(
         fileStream,
         mPatchSizeX,
         mPatchSizeY,
         mPatchSizeF,
         nxRestrictedPreBlock,
         nyRestrictedPreBlock,
         mNfPre,
         nxRestrictedPostBlock,
         nyRestrictedPostBlock,
         mNumArbors,
         mFileExtendedFlag,
         mCompressedFlag));
}

void LocalPatchWeightsFile::readInternal(double &timestamp) {
   long numValues = mWeightData->getNumValuesPerArbor();
   auto mpiBlock  = mFileManager->getMPIBlock();
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
