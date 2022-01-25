#include "LocalPatchWeightsIO.hpp"

#include "components/PatchGeometry.hpp"
#include "utils/conversions.hpp"
#include "utils/requiredConvolveMargin.hpp"
#include "utils/weight_conversions.hpp"

#include <cstdint>

namespace PV {

LocalPatchWeightsIO::LocalPatchWeightsIO(
      std::shared_ptr<FileStream> fileStream,
      int patchSizeX,
      int patchSizeY,
      int patchSizeF,
      int nxRestrictedPre,
      int nyRestrictedPre,
      int nfPre,
      int nxRestrictedPost,
      int nyRestrictedPost,
      int numArbors,
      bool fileExtendedFlag,
      bool compressedFlag) :
      mFileStream(fileStream),
      mPatchSizeX(patchSizeX), mPatchSizeY(patchSizeY), mPatchSizeF(patchSizeF),
      mNxRestrictedPre(nxRestrictedPre), mNyRestrictedPre(nyRestrictedPre),
      mNfPre(nfPre),
      mNxRestrictedPost(nxRestrictedPost), mNyRestrictedPost(nyRestrictedPost),
      mNumArbors(numArbors), mFileExtendedFlag(fileExtendedFlag), mCompressedFlag(compressedFlag) {
   FatalIf(
         fileStream and !fileStream->readable(),
         "FileStream \"%s\" is not readable and can't be used in a LocalPatchWeightsIO object.\n",
         fileStream->getFileName());

   mDataSize = static_cast<long>(mCompressedFlag ? sizeof(uint8_t) : sizeof(float));
   initializeMargins(); // initializes XMargin and YMargin
   initializeFrameSize();
   initializeNumFrames();

   if (!getFileStream()) { return; }

   // If writeable, initialize position at end of file.
   // If read-only, initialize position at beginning.
   // Users can call setFrameNumber() if something else is desired.
   if (getFileStream()->writeable()) {
      int numFrames = getNumFrames();
      setFrameNumber(numFrames);
   }
   else {
      setFrameNumber(0);
   }
}

void LocalPatchWeightsIO::calcExtremeWeights(
      WeightData const &weightDataRegion,
      int nxRestrictedRegion, int nyRestrictedRegion,
      float &minWeight, float &maxWeight) const {
   int nxExtended = nxRestrictedRegion + 2*getXMargin();
   int nyExtended = nyRestrictedRegion + 2*getYMargin();
   int status = PV_SUCCESS;
   if (nxExtended > weightDataRegion.getNumDataPatchesX()) {
      ErrorLog().printf(
            "calcExtremeWeights() called with NxRestrictedRegion = %d and required margin %d on "
            "either side, but weightDataRegion has only %d patches in the x-direction.\n",
            nxRestrictedRegion, getXMargin(), weightDataRegion.getNumDataPatchesX());
      status = PV_FAILURE;
   }
   if (nyExtended > weightDataRegion.getNumDataPatchesY()) {
      ErrorLog().printf(
            "calcExtremeWeights() called with NyRestrictedRegion = %d and required margin %d on "
            "either side, but weightDataRegion has only %d patches in the y-direction.\n",
            nyRestrictedRegion, getYMargin(), weightDataRegion.getNumDataPatchesY());
      status = PV_FAILURE;
   }
   FatalIf(status != PV_SUCCESS, "Bad arguments to calcExtremeWeights()\n");
   int nf = weightDataRegion.getNumDataPatchesF();

   auto xStartsAndStops = calcPatchStartsAndStops(
         weightDataRegion.getNumDataPatchesX(), nxRestrictedRegion,
         getNxRestrictedPre(), getNxRestrictedPost(), weightDataRegion.getPatchSizeX());
   auto yStartsAndStops = calcPatchStartsAndStops(
         weightDataRegion.getNumDataPatchesY(), nyRestrictedRegion,
         getNyRestrictedPre(), getNyRestrictedPost(), weightDataRegion.getPatchSizeY());

   minWeight = FLT_MAX;
   maxWeight = -FLT_MAX;
   for (int a = 0; a < weightDataRegion.getNumArbors(); ++a) {
      for (int y = 0; y < nyRestrictedRegion + 2*getYMargin(); ++y) {
         for (int x = 0; x < nxRestrictedRegion + 2*getXMargin(); ++x) {
            for (int f = 0; f < nf; ++f) {
               int kIndexInRegion = kIndex(
                     x, y, f, 
                     weightDataRegion.getNumDataPatchesX(),
                     weightDataRegion.getNumDataPatchesY(),
                     weightDataRegion.getNumDataPatchesF());
               float const *patchInRegion = weightDataRegion.getDataFromDataIndex(a, kIndexInRegion);
               if (getFileExtendedFlag()) {
                  // Now, we have to compare the values in the patch to minWeight and maxWeight,
                  // but only in the active part of the patch.
                  int const &xStart = xStartsAndStops[0].at(x);
                  int const &xStop  = xStartsAndStops[1].at(x);
                  int const &yStart = yStartsAndStops[0].at(y);
                  int const &yStop  = yStartsAndStops[1].at(y);
                  for (int ky = yStart; ky < yStop; ++ky) {
                     int kStart = kIndex(
                           xStart, ky, 0,
                           weightDataRegion.getPatchSizeX(),
                           weightDataRegion.getPatchSizeY(),
                           weightDataRegion.getPatchSizeF());
                     int numValuesInLine =
                        (xStop - xStart) * weightDataRegion.getPatchSizeF();
                     float const *lineStart = &patchInRegion[kStart];
                     for (int k = 0; k < numValuesInLine; ++k) {
                        float value = lineStart[k];
                        minWeight = value < minWeight ? value : minWeight;
                        maxWeight = value > maxWeight ? value : maxWeight;
                     }
                  }
               }
               else {
                  // Postweights situation; there are no shrunken patches.
                  long numValuesInPatch = weightDataRegion.getPatchSizeOverall();
                  for (long k = 0; k < numValuesInPatch; ++k) {
                     float value = patchInRegion[k];
                     minWeight = value < minWeight ? value : minWeight;
                     maxWeight = value < maxWeight ? value : maxWeight;
                  }
               }
            }
         }
      }
   }
}

long LocalPatchWeightsIO::calcFilePositionFromFrameNumber(int frameNumber) const {
   return static_cast<long>(frameNumber) * mFrameSize;
}

int LocalPatchWeightsIO::calcFrameNumberFromFilePosition(long filePosition) const {
   return static_cast<int>(filePosition / mFrameSize);
}

void LocalPatchWeightsIO::finishWrite() {
   setFrameNumber(getFrameNumber() + 1);
   if (getNumFrames() < getFrameNumber()) {
      mNumFrames = getFrameNumber();
   }
   getFileStream()->setOutPos(0L, std::ios_base::end);
   long eofPos = getFileStream()->getOutPos();
   long correctEOFPos = calcFilePositionFromFrameNumber(getNumFrames());
   if (eofPos < correctEOFPos) {
      long numPadBytes = correctEOFPos - eofPos;
      std::vector<uint8_t> nulldata(numPadBytes);
      getFileStream()->write(nulldata.data(), numPadBytes);
   }
   setFrameNumber(getFrameNumber());
}

BufferUtils::WeightHeader LocalPatchWeightsIO::readHeader() {
   BufferUtils::WeightHeader header;
   getFileStream()->read(&header, mHeaderSize);
   setFrameNumber(getFrameNumber());
   return header;
}

BufferUtils::WeightHeader LocalPatchWeightsIO::readHeader(int frameNumber) {
   setFrameNumber(frameNumber);
   return readHeader();
}

void LocalPatchWeightsIO::readRegion(
      WeightData &weightData,
      BufferUtils::WeightHeader const &header,
      int regionNxRestrictedPre,
      int regionNyRestrictedPre,
      int regionXStartRestricted,
      int regionYStartRestricted,
      int regionFStartRestricted,
      int arborIndexStart) {
   if (!mFileStream) { return; }

   if (weightData.getNumArbors() == 0) { return; }

   checkDimensions(
         weightData,
         regionNxRestrictedPre,
         regionNyRestrictedPre,
         regionXStartRestricted,
         regionYStartRestricted,
         regionFStartRestricted,
         arborIndexStart,
         std::string("readRegion"));

   std::vector<float> readBuffer(getPatchSizeOverall());
   int xMarginRegion = (weightData.getNumDataPatchesX() - regionNxRestrictedPre) / 2;
   pvAssert(xMarginRegion >= getXMargin()); // checked in checkDimensions()
   int yMarginRegion = (weightData.getNumDataPatchesY() - regionNyRestrictedPre) / 2;
   pvAssert(yMarginRegion >= getYMargin()); // checked in checkDimensions()

   auto xStartsAndStops = calcPatchStartsAndStops(
      weightData.getNumDataPatchesX(), regionNxRestrictedPre,
      getNxRestrictedPre(), getNxRestrictedPost(), getPatchSizeX());
   auto yStartsAndStops = calcPatchStartsAndStops(
      weightData.getNumDataPatchesY(), regionNyRestrictedPre,
      getNyRestrictedPre(), getNyRestrictedPost(), getPatchSizeY());

   for (int a = 0; a < weightData.getNumArbors(); ++a) {
      for (int y = -getYMargin(); y < regionNyRestrictedPre + getYMargin(); ++y) {
         for (int x = -getXMargin(); x < regionNxRestrictedPre + getXMargin(); ++x) {
            int nfRegion = weightData.getNumDataPatchesF();
            for (int f = 0; f < nfRegion; ++f) {
               int xIndexInFile = x + regionXStartRestricted + getXMargin();
               int yIndexInFile = y + regionYStartRestricted + getYMargin();
               int fIndexInFile = f + regionFStartRestricted;

               readPatch(
                     readBuffer, a, xIndexInFile, yIndexInFile, fIndexInFile,
                     header.minVal, header.maxVal);
               int xIndexInRegion = x + xMarginRegion; 
               int yIndexInRegion = y + yMarginRegion; 
               int kIndexInRegion = kIndex(
                     xIndexInRegion, yIndexInRegion, f,
                     weightData.getNumDataPatchesX(),
                     weightData.getNumDataPatchesY(),
                     weightData.getNumDataPatchesF());
               float *patchInRegion = weightData.getDataFromDataIndex(a, kIndexInRegion);
               if (getFileExtendedFlag()) {
                  // Now, we have to move the data from readBuffer.data() to patchInRegion,
                  // but only copy the active part of the patch.
                  int const &xStart = xStartsAndStops[0].at(xIndexInRegion);
                  int const &xStop  = xStartsAndStops[1].at(xIndexInRegion);
                  int const &yStart = yStartsAndStops[0].at(yIndexInRegion);
                  int const &yStop  = yStartsAndStops[1].at(yIndexInRegion);
                  for (int ky = yStart; ky < yStop; ++ky) {
                     int kStart = kIndex(
                           xStart, ky, 0, getPatchSizeX(), getPatchSizeY(), getPatchSizeF());
                     int numBytesToCopy =
                        (xStop - xStart) * getPatchSizeF() * static_cast<int>(mDataSize);
                     memcpy(&patchInRegion[kStart], &readBuffer.data()[kStart], numBytesToCopy);
                  }
               }
               else {
                  // Postweights situation; there are no shrunken patches.
                  memcpy(patchInRegion, readBuffer.data(), mDataSize * getPatchSizeOverall());
               }
            }
         }
      }
   }
   setFrameNumber(getFrameNumber());
}

void LocalPatchWeightsIO::writeHeader(BufferUtils::WeightHeader const &header) {
   getFileStream()->write(&header, mHeaderSize);
   setFrameNumber(getFrameNumber());
}

void LocalPatchWeightsIO::writeHeader(BufferUtils::WeightHeader const &header, int frameNumber) {
   setFrameNumber(frameNumber);
   writeHeader(header);
}

void LocalPatchWeightsIO::writeRegion(
      WeightData const &weightData,
      BufferUtils::WeightHeader const &header,
      int regionNxRestrictedPre,
      int regionNyRestrictedPre,
      int regionXStartRestricted,
      int regionYStartRestricted,
      int regionFStartRestricted,
      int arborIndexStart) {
   if (!mFileStream) { return; }

   if (weightData.getNumArbors() == 0) { return; }

   checkDimensions(
         weightData,
         regionNxRestrictedPre,
         regionNyRestrictedPre,
         regionXStartRestricted,
         regionYStartRestricted,
         regionFStartRestricted,
         arborIndexStart,
         std::string("writeRegion"));

   std::vector<float> writeBuffer(getPatchSizeOverall());
   int xMarginRegion = (weightData.getNumDataPatchesX() - regionNxRestrictedPre) / 2;
   pvAssert(xMarginRegion >= getXMargin()); // checked in checkDimensions()
   int yMarginRegion = (weightData.getNumDataPatchesY() - regionNyRestrictedPre) / 2;
   pvAssert(yMarginRegion >= getYMargin()); // checked in checkDimensions()

   auto xStartsAndStops = calcPatchStartsAndStops(
      weightData.getNumDataPatchesX(), regionNxRestrictedPre,
      getNxRestrictedPre(), getNxRestrictedPost(), getPatchSizeX());
   auto yStartsAndStops = calcPatchStartsAndStops(
      weightData.getNumDataPatchesY(), regionNyRestrictedPre,
      getNyRestrictedPre(), getNyRestrictedPost(), getPatchSizeY());

   for (int a = 0; a < weightData.getNumArbors(); ++a) {
      for (int y = -getYMargin(); y < regionNyRestrictedPre + getYMargin(); ++y) {
         for (int x = -getXMargin(); x < regionNxRestrictedPre + getXMargin(); ++x) {
            int nfRegion = weightData.getNumDataPatchesF();
            for (int f = 0; f < nfRegion; ++f) {
               int xIndexInFile = x + regionXStartRestricted + getXMargin();
               int yIndexInFile = y + regionYStartRestricted + getYMargin();
               int fIndexInFile = f + regionFStartRestricted;

               int xIndexInRegion = x + xMarginRegion; 
               int yIndexInRegion = y + yMarginRegion; 
               int kIndexInRegion = kIndex(
                     xIndexInRegion, yIndexInRegion, f,
                     weightData.getNumDataPatchesX(),
                     weightData.getNumDataPatchesY(),
                     weightData.getNumDataPatchesF());
               int const &xStart = xStartsAndStops[0].at(xIndexInRegion);
               int const &xStop  = xStartsAndStops[1].at(xIndexInRegion);
               int const &yStart = yStartsAndStops[0].at(yIndexInRegion);
               int const &yStop  = yStartsAndStops[1].at(yIndexInRegion);
               float const *patchInRegion = weightData.getDataFromDataIndex(a, kIndexInRegion);
               if (getFileExtendedFlag()) {
                  // Now, we have to move the data from readBuffer.data() to patchInRegion,
                  // but only copy the active part of the patch.
                  memset(writeBuffer.data(), 0, mDataSize * getPatchSizeOverall());
                  for (int ky = yStart; ky < yStop; ++ky) {
                     int kStart =
                           kIndex(xStart, ky, 0, getPatchSizeX(), getPatchSizeY(), getPatchSizeF());
                     int numBytesToCopy =
                        (xStop - xStart) * getPatchSizeF() * static_cast<int>(mDataSize);
                     memcpy(&writeBuffer.data()[kStart], &patchInRegion[kStart], numBytesToCopy);
                  }
               }
               else {
                  // Postweights situation; there are no shrunken patches.
                  memcpy(writeBuffer.data(), patchInRegion, mDataSize * getPatchSizeOverall());
               }
               // Finally, we have to write the modified patch back into the file.
               writePatch(
                     writeBuffer, a, xIndexInFile, yIndexInFile, fIndexInFile,
                     xStart, xStop, yStart, yStop, header.minVal, header.maxVal);
            }
         }
      }
   }
   setFrameNumber(getFrameNumber());
}

void LocalPatchWeightsIO::open() {
   mFileStream->open();
}

void LocalPatchWeightsIO::close() {
   mFileStream->close();
}

long LocalPatchWeightsIO::getNumPatchesFile() const {
   long nx = getNxRestrictedPre() + 2 * getXMargin();
   long ny = getNyRestrictedPre() + 2 * getYMargin();
   long nf = getNfPre();
   return (nx * ny * nf);
}

void LocalPatchWeightsIO::setFrameNumber(int frameNumber) {
   pvAssert(mFileStream);
   mFrameNumber = frameNumber;
   long filePos = calcFilePositionFromFrameNumber(frameNumber);
   mFileStream->setInPos(filePos, std::ios_base::beg);
   if (mFileStream->writeable()) { mFileStream->setOutPos(filePos, std::ios_base::beg); }
}

long LocalPatchWeightsIO::calcArborSizeBytes() const {
   long patchSizeBytes = calcPatchSizeBytes();
   long numPatches = getNumPatchesFile();
   long sizeBytes = numPatches * patchSizeBytes;
   return sizeBytes;
}

long LocalPatchWeightsIO::calcFrameSizeBytes() const {
   long sizeBytes = mHeaderSize + static_cast<long>(mNumArbors) * calcArborSizeBytes();
   return sizeBytes;
}

long LocalPatchWeightsIO::calcPatchSizeBytes() const {
   return mDataSize * getPatchSizeOverall() + mPatchHeaderSize;
}

std::array<std::vector<int>, 2> LocalPatchWeightsIO::calcPatchStartsAndStops(
      int nExtendedPre, int nRestrictedPre, int nPreRef, int nPostRef, int patchSize) {
   std::array<std::vector<int>, 2> result;
   result[0].resize(nExtendedPre);
   result[1].resize(nExtendedPre);
   
   float nPostRefFloat   = static_cast<float>(nPostRef);
   float nPreRefFloat    = static_cast<float>(nPreRef);
   float nRestrictedPreF = static_cast<float>(nRestrictedPre);
   float nPostFloat      = std::round(nPostRefFloat / nPreRefFloat * nRestrictedPreF);
   int nPost             = static_cast<int>(nPostFloat);
   int beginMargin       = (nExtendedPre - nRestrictedPre) / 2;
   int endMargin         = nExtendedPre - nRestrictedPre - beginMargin;
   int start, dim;
   int dummy1, dummy2, dummy3;
   for (int k = 0; k < nExtendedPre; ++k) {
      PatchGeometry::calcPatchData(
            k, nRestrictedPre, beginMargin, endMargin, 
            nPost, 0, 0, patchSize, &dim, &start, &dummy1, &dummy2, &dummy3);
      result[0].at(k) = start;
      result[1].at(k) = start + dim;
   }
   return result;
}

void LocalPatchWeightsIO::checkDimensions(
      WeightData const &weightData,
      int regionNxRestrictedPre,
      int regionNyRestrictedPre,
      int regionXStartRestricted,
      int regionYStartRestricted,
      int regionFStartRestricted,
      int arborIndexStart,
      std::string const &functionName) {
   std::string errMsgHdr(functionName);
   errMsgHdr.append("() for \"").append(getFileStream()->getFileName()).append("\"");

   int status = PV_SUCCESS;
   if (arborIndexStart < 0) {
      ErrorLog().printf(
            "%s called with negative arbor index (%d)\n", errMsgHdr.c_str(), arborIndexStart);
      status = PV_FAILURE;
   }
   int arborIndexStop = arborIndexStart + weightData.getNumArbors();
   if (arborIndexStop < arborIndexStart or arborIndexStop > getNumArbors()) {
      ErrorLog().printf(
            "%s called with too many arbors "
            "(region has %d arbors with starting index %d; NumArbors is %d)\n",
            errMsgHdr.c_str(), weightData.getNumArbors(), arborIndexStart, getNumArbors());
      status = PV_FAILURE;
   }
   if (regionXStartRestricted < 0) {
      ErrorLog().printf(
            "%s called with region's left edge negative (%d)\n",
            errMsgHdr.c_str(), regionXStartRestricted);
      status = PV_FAILURE;
   }
   if ( regionXStartRestricted + regionNxRestrictedPre > getNxRestrictedPre()) {
      ErrorLog().printf(
            "%s called with region's right edge beyond overall right edge (%d + %d versus %d)\n",
            errMsgHdr.c_str(), regionXStartRestricted, regionNxRestrictedPre, getNxRestrictedPre());
      status = PV_FAILURE;
   }
   if (regionYStartRestricted < 0) {
      ErrorLog().printf(
            "%s called with region's top edge negative (%d)\n",
            errMsgHdr.c_str(), regionYStartRestricted);
      status = PV_FAILURE;
   }
   if ( regionYStartRestricted + regionNyRestrictedPre > getNyRestrictedPre()) {
      ErrorLog().printf(
            "%s called with region's bottom edge beyond overall bottom edge (%d + %d versus %d)\n",
            errMsgHdr.c_str(), regionYStartRestricted, regionNyRestrictedPre, getNyRestrictedPre());
      status = PV_FAILURE;
   }
   if (regionNxRestrictedPre + 2*getXMargin() > weightData.getNumDataPatchesX()) {
      ErrorLog().printf(
            "%s called with region restricted width %d "
            "and required margin %d on each side, but WeightData only has width %d\n",
            errMsgHdr.c_str(), regionNxRestrictedPre,
            getXMargin(), weightData.getNumDataPatchesX());
      status = PV_FAILURE;
   }
   if (regionNyRestrictedPre + 2*getYMargin() > weightData.getNumDataPatchesY()) {
      ErrorLog().printf(
            "%s called with region restricted height %d "
            "and required margin %d on each side, but WeightData only has height %d\n",
            errMsgHdr.c_str(), regionNyRestrictedPre,
            getYMargin(), weightData.getNumDataPatchesY());
      status = PV_FAILURE;
   }
   if (weightData.getNumDataPatchesF() + regionFStartRestricted > getNfPre()) {
      ErrorLog().printf(
           "%s called with starting feature index %d but WeightData object has %d features "
           "and LocalPatchWeightsIO object has only %d (%d + %d versus %d)\n",
           errMsgHdr.c_str(), regionFStartRestricted,
           weightData.getNumDataPatchesF(), getNfPre(),
           weightData.getNumDataPatchesF(), regionFStartRestricted, getNfPre());
      status = PV_FAILURE;
   }
   FatalIf(status != PV_SUCCESS, "%s failed\n", errMsgHdr.c_str());
}

void LocalPatchWeightsIO::checkHeader(BufferUtils::WeightHeader const &header) const {
   int status = PV_SUCCESS;
   if (header.baseHeader.numRecords != mNumArbors) {
      ErrorLog().printf(
         "LocalPatchWeightsIO object expects %d arbors, but file has %d.\n",
         mNumArbors, header.baseHeader.numRecords);
      status = PV_FAILURE;
   }
   if (header.nxp != mPatchSizeX) {
      ErrorLog().printf(
         "LocalPatchWeightsIO object expects PatchSizeX=%d, but file has %d.\n",
         mPatchSizeX, header.nxp);
      status = PV_FAILURE;
   }
   if (header.nyp != mPatchSizeY) {
      ErrorLog().printf(
         "LocalPatchWeightsIO object expects PatchSizeX=%d, but file has %d.\n",
         mPatchSizeY, header.nyp);
      status = PV_FAILURE;
   }
   if (header.nfp != mPatchSizeF) {
      ErrorLog().printf(
         "LocalPatchWeightsIO object expects PatchSizeF=%d, but file has %d.\n",
         mPatchSizeF, header.nfp);
      status = PV_FAILURE;
   }
   if (static_cast<long>(header.numPatches) != getNumPatchesFile()) {
      ErrorLog().printf(
         "LocalPatchWeightsIO object expects %ld patches, but file has %d.\n",
         getNumPatchesFile(), header.numPatches);
      status = PV_FAILURE;
   }
   FatalIf(status != PV_SUCCESS, "checkHeader failed.\n");
}

void LocalPatchWeightsIO::initializeFrameSize() {
   mFrameSize = mHeaderSize + static_cast<long>(mNumArbors) * calcArborSizeBytes();
}

void LocalPatchWeightsIO::initializeMargins() {
   if (getFileExtendedFlag()) {
      mXMargin = requiredConvolveMargin(getNxRestrictedPre(), getNxRestrictedPost(), getPatchSizeX());
      mYMargin = requiredConvolveMargin(getNyRestrictedPre(), getNyRestrictedPost(), getPatchSizeY());
   }
   else {
      mXMargin = 0;
      mYMargin = 0;
   }
}

void LocalPatchWeightsIO::initializeNumFrames() {
   if (!getFileStream()) { return; }

   long curPos = getFileStream()->getInPos();
   getFileStream()->setInPos(0L, std::ios_base::end);
   long eofPosition = getFileStream()->getInPos();
   mNumFrames = calcFrameNumberFromFilePosition(eofPosition);
   getFileStream()->setInPos(curPos, std::ios_base::beg);
}

void LocalPatchWeightsIO::readPatch(
      std::vector<float> &readBuffer,
      int arborIndex,
      int xPatchIndex,
      int yPatchIndex,
      int fPatchIndex,
      float minWeight,
      float maxWeight) {
   pvAssert(mFileStream);
   pvAssert(static_cast<long>(readBuffer.size()) >= getPatchSizeOverall());

   int nxExtendedPre = getNxRestrictedPre() + 2 * getXMargin();
   int nyExtendedPre = getNyRestrictedPre() + 2 * getYMargin();
   int nfPre         = getNfPre();
   long patchIndexInFile = static_cast<long>(kIndex(
         xPatchIndex, yPatchIndex, fPatchIndex,
         nxExtendedPre, nyExtendedPre, nfPre));
   long patchDataOffset =
         mHeaderSize + patchIndexInFile * calcPatchSizeBytes() + mPatchHeaderSize;
   long frameStart     = calcFilePositionFromFrameNumber(getFrameNumber());
   long arborSizeBytes = calcArborSizeBytes();
   long patchOffsetBytes  = patchIndexInFile * calcPatchSizeBytes();
   long patchLocationFile =
         frameStart + mHeaderSize + arborIndex * arborSizeBytes + patchOffsetBytes;
   long patchDataLocationFile = patchLocationFile + mPatchHeaderSize;
   getFileStream()->setInPos(patchDataLocationFile, std::ios_base::beg);
   if (getCompressedFlag()) {
      long patchSizeOverall = getPatchSizeOverall();
      std::vector<uint8_t> readBufferCompressed(patchSizeOverall);
      mFileStream->read(readBufferCompressed.data(), patchSizeOverall);
      for (int k = 0; k < patchSizeOverall; ++k) {
         float compressedVal = static_cast<float>(readBufferCompressed[k])/255.0f;
         readBuffer[k] = compressedVal * (maxWeight - minWeight) + minWeight;
      }
   }
   else {
      mFileStream->read(readBuffer.data(), getPatchSizeOverall() * mDataSize);
   }
}

void LocalPatchWeightsIO::writePatch(
      std::vector<float> const &writeBuffer,
      int arborIndex,
      int xPatchIndex,
      int yPatchIndex,
      int fPatchIndex,
      int xStart,
      int xStop,
      int yStart,
      int yStop,
      float minWeight,
      float maxWeight) {
   pvAssert(mFileStream);
   pvAssert(static_cast<long>(writeBuffer.size()) >= getPatchSizeOverall());

   int nxExtendedPre = getNxRestrictedPre() + 2 * getXMargin();
   int nyExtendedPre = getNyRestrictedPre() + 2 * getYMargin();
   int nfPre         = getNfPre();
   long patchIndexInFile = static_cast<long>(kIndex(
         xPatchIndex, yPatchIndex, fPatchIndex,
         nxExtendedPre, nyExtendedPre, nfPre));
   long frameStart        = calcFilePositionFromFrameNumber(getFrameNumber());
   long arborSizeBytes    = calcArborSizeBytes();
   long patchOffsetBytes  = patchIndexInFile * calcPatchSizeBytes();
   long patchLocationFile =
         frameStart + mHeaderSize + arborIndex * arborSizeBytes + patchOffsetBytes;
   getFileStream()->setOutPos(patchLocationFile, std::ios_base::beg);
   Patch patchHeader;
   patchHeader.nx = static_cast<uint16_t>(getPatchSizeX());
   patchHeader.ny = static_cast<uint16_t>(getPatchSizeY());
   patchHeader.offset = static_cast<uint32_t>(0);
   getFileStream()->write(&patchHeader, mPatchHeaderSize);
   
   writePatchAtLocation(writeBuffer, xStart, xStop, yStart, yStop, minWeight, maxWeight);
}

void LocalPatchWeightsIO::writePatchAtLocation(
      std::vector<float> const &writeBuffer,
      int xStart, int xStop, int yStart, int yStop, float minWeight, float maxWeight) {
   pvAssert(xStart >= 0 and xStart <= xStop);
   pvAssert(xStop >= 0 and xStop <= getPatchSizeX());
   pvAssert(yStart >= 0 and yStart <= yStop);
   pvAssert(yStop >= 0 and yStop <= getPatchSizeY());
   long positionInFile = getFileStream()->getOutPos();
   if (getCompressedFlag()) {
      std::vector<uint8_t> writeBufferCompressed(getPatchSizeOverall());
      for (int k = 0; k < getPatchSizeOverall(); ++k) {
         float value         = writeBuffer[k];
         float compressedVal = (value - minWeight) / (maxWeight - minWeight) * 255.0f;
         writeBufferCompressed[k] = static_cast<uint8_t>(compressedVal);
      }
      for (int y = yStart; y < yStop; ++y) {
         int startIndex = getPatchSizeF() * (xStart + y * getPatchSizeX());
         int lineSize   = getPatchSizeF() * (xStop - xStart);
         getFileStream()->setOutPos(
               positionInFile + static_cast<long>(startIndex), std::ios_base::beg);
         mFileStream->write(writeBufferCompressed.data() + startIndex, static_cast<long>(lineSize));
      }
   }
   else {
      for (int y = yStart; y < yStop; ++y) {
         int startIndex     = getPatchSizeF() * (xStart + y * getPatchSizeX());
         long lineSizeBytes = mDataSize * static_cast<long>(getPatchSizeF() * (xStop - xStart));
         getFileStream()->setOutPos(
               positionInFile + static_cast<long>(startIndex) * mDataSize, std::ios_base::beg);
         mFileStream->write(writeBuffer.data() + startIndex, lineSizeBytes);
      }
   }
   getFileStream()->setOutPos(positionInFile, std::ios_base::beg);
}
   
} // namespace PV
