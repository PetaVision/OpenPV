#include "WeightsFileIO.hpp"
#include <cstdint>

namespace PV {

WeightsFileIO::WeightsFileIO(FileStream *fileStream, MPIBlock const *mpiBlock, Weights *weights)
      : mFileStream(fileStream), mMPIBlock(mpiBlock), mWeights(weights) {
   if (mMPIBlock == nullptr) {
      throw std::invalid_argument("WeightsFileIO instantiated with a null MPIBlock");
   }
   if (mMPIBlock->getRank() == mRootProcess and mFileStream == nullptr) {
      throw std::invalid_argument(
            "WeightsFileIO instantiated with a null file stream on the null process");
   }
   if (mWeights == nullptr) {
      throw std::invalid_argument("WeightsFileIO instantiated with a null Weights object");
   }
}

// function members for reading
double WeightsFileIO::readWeights(int frameNumber) {
   if (mFileStream != nullptr and !mFileStream->readable()) {
      throw std::invalid_argument(
            "WeightsFileIO::readWeights called with a nonreadable file stream");
   }
   BufferUtils::WeightHeader header = readHeader(frameNumber);
   checkHeader(header);

   double timestamp;
   if (mWeights->getSharedFlag()) {
      timestamp = readSharedWeights(frameNumber, header);
   }
   else {
      timestamp = readNonsharedWeights(frameNumber, header);
   }
   mWeights->setTimestamp(timestamp);
   return timestamp;
}

BufferUtils::WeightHeader WeightsFileIO::readHeader(int frameNumber) {
   BufferUtils::WeightHeader header;
   int const rank = mMPIBlock->getRank();
   if (rank == mRootProcess) {
      moveToFrame(header, *mFileStream, frameNumber);
   }

   MPI_Bcast(&header, (int)sizeof(header), MPI_BYTE, mRootProcess, mMPIBlock->getComm());
   return header;
}

void WeightsFileIO::checkHeader(BufferUtils::WeightHeader const &header) {
   if (mWeights->getSharedFlag()) {
      FatalIf(
            header.baseHeader.fileType != PVP_KERNEL_FILE_TYPE,
            "Connection \"%s\" has sharedWeights true, ",
            "but \"%s\" is not a shared-weights file\n",
            mWeights->getName().c_str(),
            mFileStream->getFileName().c_str());
      FatalIf(
            header.numPatches != mWeights->getNumDataPatches(),
            "Shared-weights connection \"%s\" has a unit cell (%d-by-%d-by-%d), "
            "but \"%s\" has %d patches.\n",
            mWeights->getName().c_str(),
            mWeights->getNumDataPatchesX(),
            mWeights->getNumDataPatchesY(),
            mWeights->getNumDataPatchesF(),
            mFileStream->getFileName().c_str(),
            header.numPatches);
   }
   else {
      // TODO: It should be allowed to read a kernel file into a non-shared-weights atlas
      FatalIf(
            header.baseHeader.fileType != PVP_WGT_FILE_TYPE,
            "Connection \"%s\" has sharedWeights false.\n",
            "but \"%s\" is not a non-shared-weights file. ",
            mWeights->getName().c_str(),
            mFileStream->getFileName().c_str());
   }
   FatalIf(
         header.baseHeader.nBands < mWeights->getNumArbors(),
         "Connection \"%s\" has %d arbors, but file \"%s\" has only %d arbors.\n",
         mWeights->getName().c_str(),
         mWeights->getNumArbors(),
         mFileStream->getFileName().c_str(),
         header.baseHeader.nBands);

   FatalIf(
         header.nxp != mWeights->getPatchSizeX(),
         "Connection \"%s\" has nxp=%d, but file \"%s\" has nxp=%d.\n",
         mWeights->getName().c_str(),
         mWeights->getPatchSizeX(),
         mFileStream->getFileName().c_str(),
         header.nxp);
   FatalIf(
         header.nyp != mWeights->getPatchSizeY(),
         "Connection \"%s\" has nyp=%d, but file \"%s\" has nyp=%d.\n",
         mWeights->getName().c_str(),
         mWeights->getPatchSizeY(),
         mFileStream->getFileName().c_str(),
         header.nyp);
   FatalIf(
         header.nfp != mWeights->getPatchSizeF(),
         "Connection \"%s\" has nfp=%d, but file \"%s\" has nfp=%d.\n",
         mWeights->getName().c_str(),
         mWeights->getPatchSizeF(),
         mFileStream->getFileName().c_str(),
         header.nfp);
}

bool WeightsFileIO::isCompressedHeader(BufferUtils::WeightHeader const &header) {
   bool isCompressed;
   switch (header.baseHeader.dataType) {
      case BufferUtils::BYTE:
         FatalIf(
               header.baseHeader.dataSize != (int)sizeof(unsigned char),
               "File \"%s\" has dataSize=%d, inconsistent with dataType BYTE (%d)\n",
               mFileStream->getFileName().c_str(),
               header.baseHeader.dataSize,
               header.baseHeader.dataType);
         isCompressed = true;
         break;
      case BufferUtils::FLOAT:
         FatalIf(
               header.baseHeader.dataSize != (int)sizeof(float),
               "File \"%s\" has dataSize=%d, inconsistent with dataType FLOAT (%d)\n",
               mFileStream->getFileName().c_str(),
               header.baseHeader.dataSize,
               header.baseHeader.dataType);
         isCompressed = false;
         break;
      case BufferUtils::INT:
         Fatal().printf(
               "File \"%s\" has dataType INT. Only FLOAT and BYTE are supported.\n",
               mFileStream->getFileName().c_str());
         break;
      default:
         Fatal().printf(
               "File \"%s\" has unrecognized datatype.\n", mFileStream->getFileName().c_str());
         break;
   }
   return isCompressed;
}

double WeightsFileIO::readSharedWeights(int frameNumber, BufferUtils::WeightHeader const &header) {
   bool compressed          = isCompressedHeader(header);
   double timestamp         = header.baseHeader.timestamp;
   long arborSizeInPvpFile  = calcArborSizeLocal(compressed);
   long arborSizeInPvpLocal = arborSizeInPvpFile;
   std::vector<unsigned char> readBuffer(arborSizeInPvpLocal);

   int const numArbors = mWeights->getNumArbors();
   for (int arbor = 0; arbor < numArbors; arbor++) {
      if (mMPIBlock->getRank() == mRootProcess) {
         mFileStream->read(readBuffer.data(), arborSizeInPvpFile);
      }
      MPI_Bcast(
            readBuffer.data(), arborSizeInPvpFile, MPI_BYTE, mRootProcess, mMPIBlock->getComm());
      loadWeightsFromBuffer(readBuffer, arbor, header.minVal, header.maxVal, compressed);
   }
   return timestamp;
}

double
WeightsFileIO::readNonsharedWeights(int frameNumber, BufferUtils::WeightHeader const &header) {
   bool compressed          = isCompressedHeader(header);
   long arborSizeInPvpFile  = calcArborSizeFile(compressed);
   long arborSizeInPvpLocal = calcArborSizeLocal(compressed);
   std::vector<unsigned char> readBuffer(arborSizeInPvpLocal);

   int const nxp           = mWeights->getPatchSizeX();
   int const nyp           = mWeights->getPatchSizeY();
   int const nfp           = mWeights->getPatchSizeF();
   long patchSizePvpFormat = (long)BufferUtils::weightPatchSize(nxp * nyp * nfp, compressed);

   int const numArbors = mWeights->getNumArbors();
   if (mMPIBlock->getRank() == mRootProcess) {
      long const frameStartFile = mFileStream->getInPos();
      for (int arbor = 0; arbor < numArbors; arbor++) {
         long const arborStartInFile = frameStartFile + (long)(arbor * arborSizeInPvpFile);
         mFileStream->setInPos(arborStartInFile, true /*from beginning of file*/);

         // For each process, need to determine patches to load from the PVP file.
         // The patch atlas may have a bigger border than the PVP file.
         int startPatchX, endPatchX, startPatchY, endPatchY;
         calcPatchBox(startPatchX, endPatchX, startPatchY, endPatchY);
         int lineCount = (endPatchX - startPatchX) * mWeights->getNumDataPatchesF();

         PVLayerLoc const &preLoc  = mWeights->getGeometry()->getPreLoc();
         PVLayerLoc const &postLoc = mWeights->getGeometry()->getPostLoc();

         int marginX    = calcNeededBorder(preLoc.nx, postLoc.nx, nxp);
         int nxExtended = preLoc.nx * mMPIBlock->getGlobalNumColumns() + marginX + marginX;

         int marginY    = calcNeededBorder(preLoc.ny, postLoc.ny, nyp);
         int nyExtended = preLoc.ny * mMPIBlock->getGlobalNumRows() + marginY + marginY;

         for (int destRank = 0; destRank < mMPIBlock->getSize(); destRank++) {
            int rowIndex, columnIndex, batchElemIndex;
            mMPIBlock->calcRowColBatchFromRank(destRank, rowIndex, columnIndex, batchElemIndex);

            for (int y = 0; y < endPatchY - startPatchY; y++) {
               int const startFileX = columnIndex * preLoc.nx;
               int const startFileY = y + rowIndex * preLoc.ny;
               int const startFile  = kIndex(
                     startFileX, startFileY, 0, nxExtended, nyExtended, header.baseHeader.nf);
               long lineStartInFile = arborStartInFile + (long)startFile * patchSizePvpFormat;
               mFileStream->setInPos(lineStartInFile, true /*from beginning of file*/);

               int const startPatchLocal = kIndex(
                     startPatchX,
                     y + startPatchY,
                     0,
                     mWeights->getNumDataPatchesX(),
                     mWeights->getNumDataPatchesY(),
                     mWeights->getNumDataPatchesF());

               unsigned char *lineLocInBuffer =
                     &readBuffer[(long)startPatchLocal * patchSizePvpFormat];
               std::size_t bufferSize = (std::size_t)lineCount * (std::size_t)patchSizePvpFormat;
               mFileStream->read(lineLocInBuffer, bufferSize);
            }
            if (destRank == mRootProcess) {
               loadWeightsFromBuffer(readBuffer, arbor, header.minVal, header.maxVal, compressed);
            }
            else {
               int tag       = tagbase + arbor;
               MPI_Comm comm = mMPIBlock->getComm();
               MPI_Send(readBuffer.data(), (int)readBuffer.size(), MPI_BYTE, destRank, tag, comm);
            }
         }
      }
   }
   else {
      for (int arbor = 0; arbor < numArbors; arbor++) {
         int tag       = tagbase + arbor;
         MPI_Comm comm = mMPIBlock->getComm();
         MPI_Recv(
               readBuffer.data(),
               (int)readBuffer.size(),
               MPI_BYTE,
               mRootProcess,
               tag,
               comm,
               MPI_STATUS_IGNORE);
         loadWeightsFromBuffer(readBuffer, arbor, header.minVal, header.maxVal, compressed);
      }
   }
   return header.baseHeader.timestamp;
}

// function members for writing
void WeightsFileIO::writeWeights(double timestamp, bool compress) {
   if (mFileStream != nullptr and !mFileStream->writeable()) {
      throw std::invalid_argument(
            "WeightsFileIO::writeWeights called with a nonwriteable file stream");
   }
   if (mWeights->getSharedFlag()) {
      writeSharedWeights(timestamp, compress);
   }
   else {
      writeNonsharedWeights(timestamp, compress);
   }
}

void WeightsFileIO::writeSharedWeights(double timestamp, bool compress) {
   if (mMPIBlock->getRank() != mRootProcess) {
      return;
   }
   float minWeight                  = mWeights->calcMinWeight();
   float maxWeight                  = mWeights->calcMaxWeight();
   BufferUtils::WeightHeader header = BufferUtils::buildSharedWeightHeader(
         mWeights->getPatchSizeX(),
         mWeights->getPatchSizeY(),
         mWeights->getPatchSizeF(),
         mWeights->getNumArbors(),
         mWeights->getNumDataPatchesX(),
         mWeights->getNumDataPatchesY(),
         mWeights->getNumDataPatchesF(),
         timestamp,
         compress,
         minWeight,
         maxWeight);

   mFileStream->write(&header, sizeof(header));

   long arborSizeInPvpFile  = calcArborSizeLocal(compress);
   long arborSizeInPvpLocal = arborSizeInPvpFile;
   std::vector<unsigned char> writeBuffer(arborSizeInPvpLocal);

   int const numArbors = mWeights->getNumArbors();
   for (int arbor = 0; arbor < numArbors; arbor++) {
      storeSharedPatches(writeBuffer, arbor, minWeight, maxWeight, compress);
      mFileStream->write(writeBuffer.data(), arborSizeInPvpFile);
   }
}

void WeightsFileIO::writeNonsharedWeights(double timestamp, bool compress) {
   float extrema[2];
   extrema[0] = mWeights->calcMinWeight();
   extrema[1] = -mWeights->calcMaxWeight();
   MPI_Allreduce(MPI_IN_PLACE, extrema, 2, MPI_FLOAT, MPI_MIN, mMPIBlock->getComm());
   extrema[1] = -extrema[1];

   long arborSizeInPvpFile  = calcArborSizeFile(compress);
   long arborSizeInPvpLocal = calcArborSizeLocal(compress);
   std::vector<unsigned char> writeBuffer(arborSizeInPvpLocal);

   int const numArbors = mWeights->getNumArbors();
   if (mMPIBlock->getRank() == mRootProcess) {

      BufferUtils::WeightHeader header = BufferUtils::buildNonsharedWeightHeader(
            mWeights->getPatchSizeX(),
            mWeights->getPatchSizeY(),
            mWeights->getPatchSizeF(),
            mWeights->getNumArbors(),
            true /*extended*/,
            timestamp,
            &mWeights->getGeometry()->getPreLoc(),
            &mWeights->getGeometry()->getPostLoc(),
            mMPIBlock->getNumColumns(),
            mMPIBlock->getNumRows(),
            extrema[0] /*min weight*/,
            extrema[1] /*max weight*/,
            compress);
      mFileStream->write(&header, sizeof(header));

      long const frameStartFile = mFileStream->getOutPos();
      for (int arbor = 0; arbor < numArbors; arbor++) {
         long const arborStartFile = frameStartFile + (long)(arbor * arborSizeInPvpFile);
         mFileStream->setOutPos(arborStartFile, true /*from beginning of file*/);

         // For each process, need to determine patches to write to the PVP file.
         // The patch atlas may have a bigger border than the PVP file has.
         int startPatchX, endPatchX, startPatchY, endPatchY;
         calcPatchBox(startPatchX, endPatchX, startPatchY, endPatchY);
         int const numDataPatchesF = mWeights->getNumDataPatchesF();
         int startPatchK           = startPatchX * numDataPatchesF;
         int endPatchK             = endPatchX * numDataPatchesF;
         int const numDataPatchesK = mWeights->getNumDataPatchesX() * numDataPatchesF;

         int const nxp                 = mWeights->getPatchSizeX();
         int const nyp                 = mWeights->getPatchSizeY();
         int const nfp                 = mWeights->getPatchSizeF();
         auto const patchSizePvpFormat = BufferUtils::weightPatchSize(nxp * nyp * nfp, compress);

         for (int sourceRank = 0; sourceRank < mMPIBlock->getSize(); sourceRank++) {
            int rowIndex, columnIndex, batchElemIndex;
            mMPIBlock->calcRowColBatchFromRank(sourceRank, rowIndex, columnIndex, batchElemIndex);

            if (sourceRank == mRootProcess) {
               storeNonsharedPatches(writeBuffer, arbor, extrema[0], extrema[1], compress);
            }
            else {
               int tag       = tagbase + arbor;
               MPI_Comm comm = mMPIBlock->getComm();
               MPI_Recv(
                     writeBuffer.data(),
                     (int)writeBuffer.size(),
                     MPI_BYTE,
                     sourceRank,
                     tag,
                     comm,
                     MPI_STATUS_IGNORE);
            }

            for (int y = 0; y < endPatchY - startPatchY; y++) {
               PVLayerLoc const &preLoc = mWeights->getGeometry()->getPreLoc();
               int const startFileX     = columnIndex * preLoc.nx;
               int const startFileY     = y + rowIndex * preLoc.ny;
               int const startFile      = kIndex(
                     startFileX,
                     startFileY,
                     0,
                     header.baseHeader.nxExtended,
                     header.baseHeader.nyExtended,
                     header.baseHeader.nf);
               long lineStartFile = arborStartFile + (long)startFile * (long)patchSizePvpFormat;
               mFileStream->setOutPos(lineStartFile, true /*from beginning of file*/);

               int const startPatchLocal = kIndex(
                     startPatchX,
                     y + startPatchY,
                     0,
                     mWeights->getNumDataPatchesX(),
                     mWeights->getNumDataPatchesY(),
                     mWeights->getNumDataPatchesF());

               for (int k = startPatchK; k < endPatchK; k++) {
                  int patchIndexLocal = kIndex(
                        k, y + startPatchY, 0, numDataPatchesK, mWeights->getNumDataPatchesY(), 1);
                  unsigned char *patchLocInBuffer =
                        &writeBuffer[patchIndexLocal * patchSizePvpFormat];
                  writePatch(patchLocInBuffer, compress);
               }
            }
         }
      }
      // If file length is shorter than it should be, the last patch is shrunken at the end.
      // In this case, we need to pad out the file length so that file reading does not hit
      // end-of-file too early.
      // If file length is longer than required by this frame, we don't need to do anything. This
      // situation can arise, for example, for the outputPath file from a connection if we restart
      // from a checkpoint when several frames were written after that checkpoint.
      long const frameEndFile = frameStartFile + (long)(numArbors * arborSizeInPvpFile);
      mFileStream->setOutPos(0L, std::ios_base::end);
      long const endOfFile = mFileStream->getOutPos();
      if (endOfFile < frameEndFile) {
         mFileStream->setOutPos(frameEndFile - 1L, true /*from beginning*/);
         mFileStream->write("\0", 1L);
      }
      mFileStream->setOutPos(frameEndFile, true /*from beginning*/);
   }
   else {
      for (int arbor = 0; arbor < numArbors; arbor++) {
         storeNonsharedPatches(writeBuffer, arbor, extrema[0], extrema[1], compress);
         int tag       = tagbase + arbor;
         MPI_Comm comm = mMPIBlock->getComm();
         MPI_Send(writeBuffer.data(), (int)writeBuffer.size(), MPI_BYTE, mRootProcess, tag, comm);
      }
   }
}

void WeightsFileIO::writePatch(unsigned char const *patchBuffer, bool compressed) {
   int const nxp = mWeights->getPatchSizeX();
   int const nyp = mWeights->getPatchSizeY();
   int const nfp = mWeights->getPatchSizeF();

   Patch patch;

   // In the file, patch header is always unshrunken. Otherwise, we would have to
   // handle patches in overlap regions by reading in, forming the union of active regions,
   // and then writing back. The appearance of the patch headers in the pvp file is a legacy
   // from olden times when each process wrote its own pvp file.
   patch.nx     = (std::uint16_t)nxp;
   patch.ny     = (std::uint16_t)nyp;
   patch.offset = (std::uint32_t)0;
   mFileStream->write(&patch.nx, sizeof(patch.nx));
   mFileStream->write(&patch.ny, sizeof(patch.ny));
   mFileStream->write(&patch.offset, sizeof(patch.offset));

   // Now load the patch header
   memcpy(&patch.nx, patchBuffer, sizeof(patch.nx));
   memcpy(&patch.ny, &patchBuffer[sizeof(patch.nx)], sizeof(patch.ny));
   memcpy(&patch.offset, &patchBuffer[sizeof(patch.nx) + sizeof(patch.ny)], sizeof(patch.offset));

   std::size_t patchHeaderSize         = sizeof(patch.nx) + sizeof(patch.ny) + sizeof(patch.offset);
   std::size_t dataSize                = compressed ? sizeof(unsigned char) : sizeof(float);
   std::size_t patchDataStartOffset    = patchHeaderSize + (std::size_t)patch.offset * dataSize;
   unsigned char const *patchDataStart = &patchBuffer[patchDataStartOffset];
   long patchStartInFile               = mFileStream->getOutPos();
   long patchEndInFile                 = patchStartInFile + (long)(nxp * nyp * nfp * (int)dataSize);
   mFileStream->setOutPos((long)patch.offset * (long)dataSize, false /*from current position*/);
   if ((int)patch.nx == nxp) {
      // active region is contiguous in memory; write all lines at once
      long dataLength = (long)patch.ny * (long)(nxp * nfp) * (long)dataSize;
      mFileStream->write(patchDataStart, dataLength);
   }
   else {
      // active region is not contiguous. Write each line, then skip to the start of the next line
      std::size_t stride     = (std::size_t)(nfp * nxp) * dataSize;
      std::size_t lineLength = (std::size_t)nfp * (std::size_t)patch.nx * dataSize;
      std::size_t skipLength = stride - lineLength;
      for (std::uint16_t y = (std::uint16_t)0; y < patch.ny - (std::uint16_t)1; y++) {
         unsigned char const *lineStartInBuffer = &patchDataStart[y * stride];
         mFileStream->write(lineStartInBuffer, lineLength);
         mFileStream->setOutPos(skipLength, false /*from current position*/);
      }
      if (patch.ny > (std::uint16_t)0) {
         std::size_t lastLineOffset             = (std::size_t)(patch.ny - 1) * stride;
         unsigned char const *lineStartInBuffer = &patchDataStart[lastLineOffset];
         mFileStream->write(lineStartInBuffer, lineLength);
      }
   }
   mFileStream->setOutPos(patchEndInFile, true /*from start of file*/);
}

// utility function members

void WeightsFileIO::moveToFrame(
      BufferUtils::WeightHeader &header,
      FileStream &fileStream,
      int frameNumber) {
   fileStream.setInPos(0L, true /*from beginning*/);
   for (int f = 0; f < frameNumber; f++) {
      fileStream.read(&header, sizeof(header));
      long recordSize = (long)(header.baseHeader.recordSize * header.baseHeader.numRecords);
      fileStream.setInPos(recordSize, false /*relative to current point*/);
   }
   fileStream.read(&header, sizeof(header));
}

long WeightsFileIO::calcArborSizeFile(bool compressed) {
   int const nxp       = mWeights->getPatchSizeX();
   int const nyp       = mWeights->getPatchSizeY();
   int const nfp       = mWeights->getPatchSizeF();
   int const patchSize = (int)BufferUtils::weightPatchSize(nxp * nyp * nfp, compressed);

   int numPatches;
   if (mWeights->getSharedFlag()) {
      numPatches = mWeights->getNumDataPatches();
   }
   else {
      PVLayerLoc const &preLoc  = mWeights->getGeometry()->getPreLoc();
      PVLayerLoc const &postLoc = mWeights->getGeometry()->getPostLoc();

      int marginX     = calcNeededBorder(preLoc.nx, postLoc.nx, mWeights->getPatchSizeX());
      int numPatchesX = preLoc.nx * mMPIBlock->getGlobalNumColumns() + marginX + marginX;

      int marginY     = calcNeededBorder(preLoc.ny, postLoc.ny, mWeights->getPatchSizeY());
      int numPatchesY = preLoc.ny * mMPIBlock->getGlobalNumRows() + marginY + marginY;

      numPatches = numPatchesX * numPatchesY * preLoc.nf;
   }

   int const arborSize = numPatches * patchSize;
   return arborSize;
}

long WeightsFileIO::calcArborSizeLocal(bool compressed) {
   int const nxp       = mWeights->getPatchSizeX();
   int const nyp       = mWeights->getPatchSizeY();
   int const nfp       = mWeights->getPatchSizeF();
   int const patchSize = (int)BufferUtils::weightPatchSize(nxp * nyp * nfp, compressed);

   int numPatches = mWeights->getNumDataPatches();

   int const arborSize = numPatches * patchSize;

   return arborSize;
}

void WeightsFileIO::calcPatchBox(
      int &startPatchX,
      int &endPatchX,
      int &startPatchY,
      int &endPatchY) {
   PVLayerLoc const &preLoc  = mWeights->getGeometry()->getPreLoc();
   PVLayerLoc const &postLoc = mWeights->getGeometry()->getPostLoc();
   PVHalo const &preHalo     = preLoc.halo;

   int nxp = mWeights->getPatchSizeX();
   calcPatchRange(preLoc.nx, postLoc.nx, preHalo.lt, preHalo.rt, nxp, startPatchX, endPatchX);

   int nyp = mWeights->getPatchSizeY();
   calcPatchRange(preLoc.ny, postLoc.ny, preHalo.up, preHalo.dn, nyp, startPatchY, endPatchY);
}

void WeightsFileIO::calcPatchRange(
      int nPre,
      int nPost,
      int preStartBorder,
      int preEndBorder,
      int patchSize,
      int &startPatch,
      int &endPatch) {
   int const neededBorder = calcNeededBorder(nPre, nPost, patchSize);

   startPatch = (preStartBorder >= neededBorder) ? preStartBorder - neededBorder : 0;
   endPatch   = preStartBorder + nPre;
   endPatch += (preEndBorder >= neededBorder) ? neededBorder : preEndBorder;
}

int WeightsFileIO::calcNeededBorder(int nPre, int nPost, int patchSize) {
   int neededBorder;
   if (nPre > nPost) {
      pvAssert(nPre % nPost == 0);
      int stride = nPre / nPost;
      pvAssert(stride % 2 == 0);
      int halfstride = stride / 2;
      neededBorder   = (patchSize - 1) * halfstride;
   }
   else if (nPre < nPost) {
      pvAssert(nPost % nPre == 0);
      int tstride = nPost / nPre;
      pvAssert(patchSize % tstride == 0);
      neededBorder = patchSize / (2 * tstride); // integer division
   }
   else {
      pvAssert(nPre == nPost);
      pvAssert(patchSize % 2 == 1);
      neededBorder = (patchSize - 1) / 2;
   }
   return neededBorder;
}

void WeightsFileIO::loadWeightsFromBuffer(
      std::vector<unsigned char> const &dataFromFile,
      int arbor,
      float minValue,
      float maxValue,
      bool compressed) {
   int const nxp        = mWeights->getPatchSizeX();
   int const nyp        = mWeights->getPatchSizeY();
   int const nfp        = mWeights->getPatchSizeF();
   int const numPatches = mWeights->getNumDataPatches();

   auto const patchSizePvpFormat     = BufferUtils::weightPatchSize(nxp * nyp * nfp, compressed);
   std::size_t const patchHeaderSize = sizeof(unsigned int) + 2UL * sizeof(unsigned short);
   if (compressed) {
      for (int k = 0; k < numPatches; k++) {
         std::size_t const offsetInFile     = patchSizePvpFormat * (std::size_t)k;
         unsigned char const *patchFromFile = &dataFromFile[offsetInFile + patchHeaderSize];
         float *weightsInPatch              = mWeights->getDataFromDataIndex(arbor, k);
         decompressPatch(patchFromFile, weightsInPatch, nxp * nyp * nfp, minValue, maxValue);
      }
   }
   else {
      for (int k = 0; k < numPatches; k++) {
         std::size_t const offsetInFile     = patchSizePvpFormat * (std::size_t)k;
         unsigned char const *patchFromFile = &dataFromFile[offsetInFile + patchHeaderSize];
         float *weightsInPatch              = mWeights->getDataFromDataIndex(arbor, k);
         memcpy(weightsInPatch, patchFromFile, (std::size_t)(nxp * nyp * nfp) * sizeof(float));
      }
   }
}

void WeightsFileIO::decompressPatch(
      unsigned char const *dataFromFile,
      float *destWeights,
      int count,
      float minValue,
      float maxValue) {
   for (int k = 0; k < count; k++) {
      float compressedWeight = (float)dataFromFile[k] / 255.0f;
      destWeights[k]         = (compressedWeight) * (maxValue - minValue) + minValue;
   }
}

// TODO: templating to reduce code duplication between and within store{Nonshared,Shared}Patches
void WeightsFileIO::storeSharedPatches(
      std::vector<unsigned char> &dataFromFile,
      int arbor,
      float minValue,
      float maxValue,
      bool compressed) {
   int const nxp = mWeights->getPatchSizeX();
   int const nyp = mWeights->getPatchSizeY();
   int const nfp = mWeights->getPatchSizeF();

   int const numDataPatches          = mWeights->getNumDataPatches();
   auto const patchSizePvpFormat     = BufferUtils::weightPatchSize(nxp * nyp * nfp, compressed);
   std::size_t const patchHeaderSize = sizeof(unsigned int) + 2UL * sizeof(unsigned short);
   if (compressed) {
      for (int k = 0; k < numDataPatches; k++) {
         std::size_t const offsetInFile = patchSizePvpFormat * (std::size_t)k;
         unsigned char *patchFromFile   = &dataFromFile[offsetInFile];
         unsigned short shortDim;
         shortDim = (unsigned short)nxp;
         memcpy(patchFromFile, &shortDim, sizeof(shortDim));
         shortDim = (unsigned short)nyp;
         memcpy(&patchFromFile[sizeof(shortDim)], &shortDim, sizeof(shortDim));

         // always zero offset for shared
         memset(&patchFromFile[2UL * sizeof(shortDim)], 0, sizeof(unsigned int));
         patchFromFile += patchHeaderSize;
         float const *weightsInPatch = mWeights->getDataFromDataIndex(arbor, k);
         compressPatch(patchFromFile, weightsInPatch, nxp * nyp * nfp, minValue, maxValue);
      }
   }
   else {
      for (int k = 0; k < numDataPatches; k++) {
         std::size_t const offsetInFile = patchSizePvpFormat * (std::size_t)k;
         unsigned char *patchFromFile   = &dataFromFile[offsetInFile];
         unsigned short shortDim;
         shortDim = (unsigned short)nxp;
         memcpy(patchFromFile, &shortDim, sizeof(shortDim));
         shortDim = (unsigned short)nyp;
         memcpy(&patchFromFile[sizeof(shortDim)], &shortDim, sizeof(shortDim));

         // always zero offset for shared
         memset(&patchFromFile[2UL * sizeof(shortDim)], 0, sizeof(unsigned int));
         patchFromFile += patchHeaderSize;
         float *weightsInPatch = mWeights->getDataFromDataIndex(arbor, k);
         memcpy(patchFromFile, weightsInPatch, (std::size_t)(nxp * nyp * nfp) * sizeof(float));
      }
   }
}

void WeightsFileIO::storeNonsharedPatches(
      std::vector<unsigned char> &dataFromFile,
      int arbor,
      float minValue,
      float maxValue,
      bool compressed) {
   int const nxp            = mWeights->getPatchSizeX();
   int const nyp            = mWeights->getPatchSizeY();
   int const nfp            = mWeights->getPatchSizeF();
   int const numDataPatches = mWeights->getNumDataPatches();

   auto const patchSizePvpFormat     = BufferUtils::weightPatchSize(nxp * nyp * nfp, compressed);
   std::size_t const patchHeaderSize = sizeof(std::uint32_t) + 2UL * sizeof(std::uint16_t);
   if (compressed) {
      for (int k = 0; k < numDataPatches; k++) {
         std::size_t const offsetInFile = patchSizePvpFormat * (std::size_t)k;
         unsigned char *patchFromFile   = &dataFromFile[offsetInFile];

         Patch const &patch = mWeights->getPatch(k);
         std::uint16_t shortDim;
         shortDim = (std::uint16_t)patch.nx;
         memcpy(patchFromFile, &shortDim, sizeof(shortDim));
         shortDim = (std::uint16_t)patch.ny;
         memcpy(&patchFromFile[sizeof(shortDim)], &shortDim, sizeof(shortDim));
         std::uint32_t offset = (std::uint32_t)patch.offset;
         memcpy(&patchFromFile[2UL * sizeof(shortDim)], &offset, sizeof(offset));
         patchFromFile += patchHeaderSize;
         float const *weightsInPatch = mWeights->getDataFromDataIndex(arbor, k);
         compressPatch(patchFromFile, weightsInPatch, nxp * nyp * nfp, minValue, maxValue);
      }
   }
   else {
      for (int k = 0; k < numDataPatches; k++) {
         std::size_t const offsetInFile = patchSizePvpFormat * (std::size_t)k;
         unsigned char *patchFromFile   = &dataFromFile[offsetInFile];

         Patch const &patch = mWeights->getPatch(k);
         std::uint16_t shortDim;
         shortDim = (std::uint16_t)patch.nx;
         memcpy(patchFromFile, &shortDim, sizeof(shortDim));
         shortDim = (std::uint16_t)patch.ny;
         memcpy(&patchFromFile[sizeof(shortDim)], &shortDim, sizeof(shortDim));
         std::uint32_t offset = (std::uint32_t)patch.offset;
         memcpy(&patchFromFile[2UL * sizeof(shortDim)], &offset, sizeof(offset));
         patchFromFile += patchHeaderSize;
         float const *weightsInPatch = mWeights->getDataFromDataIndex(arbor, k);
         memcpy(patchFromFile, weightsInPatch, (std::size_t)(nxp * nyp * nfp) * sizeof(float));
      }
   }
}

void WeightsFileIO::compressPatch(
      unsigned char *dataForFile,
      float const *sourceWeights,
      int count,
      float minValue,
      float maxValue) {
   for (int k = 0; k < count; k++) {
      float compressedWeight = (sourceWeights[k] - minValue) / (maxValue - minValue);
      dataForFile[k]         = (unsigned char)std::floor(255.0f * compressedWeight);
   }
}

} // namespace PV
