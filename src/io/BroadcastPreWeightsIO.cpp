#include "BroadcastPreWeightsIO.hpp"

#include "structures/PatchGeometry.hpp"
#include "include/pv_common.h"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include "utils/conversions.hpp"
#include "utils/requiredConvolveMargin.hpp"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ios>

namespace PV {

BroadcastPreWeightsIO::BroadcastPreWeightsIO(
      std::shared_ptr<FileStream> fileStream,
      int patchSizeX,
      int patchSizeY,
      int patchSizeF,
      int nfPre,
      int numArbors,
      bool compressedFlag)
      : mFileStream(fileStream),
        mPatchSizeX(patchSizeX),
        mPatchSizeY(patchSizeY),
        mPatchSizeF(patchSizeF),
        mNfPre(nfPre),
        mNumArbors(numArbors),
        mCompressedFlag(compressedFlag) {
   FatalIf(
         fileStream and !fileStream->readable(),
         "FileStream \"%s\" is not readable and can't be used in a BroadcastPreWeightsIO object.\n",
         fileStream->getFileName().c_str());

   mDataSize = static_cast<long>(mCompressedFlag ? sizeof(uint8_t) : sizeof(float));
   initializeFrameSize();
   initializeHeader();
   initializeNumFrames();

   if (!getFileStream()) {
      return;
   }

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

long BroadcastPreWeightsIO::calcFilePositionFromFrameNumber(int frameNumber) const {
   return static_cast<long>(frameNumber) * mFrameSize;
}

int BroadcastPreWeightsIO::calcFrameNumberFromFilePosition(long filePosition) const {
   return static_cast<int>(filePosition / mFrameSize);
}

   /*
    * Check region for readRegion and writeRegion:
    *     xStart + nxpLocal should be <= mHeader.nxp
    *     yStart + nypLocal should be <= mHeader.nyp
    *     fStart + nfpLocal should be <= mHeader.nfp
    *     mHeader.baseHeader.nx{,Extended} and weightData.getPatchSizeX() should each be 1
    *     mHeader.baseHeader.ny{,Extended} and weightData.getPatchSizeY() should each be 1
    *     fPreStart + nfPreLocal should be <= mHeader.baseHeader.nf
    *     arborIndexStart + numArbors should be <= mHeader.nbands
    */

void BroadcastPreWeightsIO::finishWrite() {
   setFrameNumber(getFrameNumber() + 1);
   if (getNumFrames() < getFrameNumber()) {
      mNumFrames = getFrameNumber();
   }
   getFileStream()->setOutPos(0L, std::ios_base::end);
   long eofPos        = getFileStream()->getOutPos();
   long correctEOFPos = calcFilePositionFromFrameNumber(getNumFrames());
   if (eofPos < correctEOFPos) {
      long numPadBytes = correctEOFPos - eofPos;
      std::vector<uint8_t> nulldata(numPadBytes);
      getFileStream()->write(nulldata.data(), numPadBytes);
   }
   setFrameNumber(getFrameNumber());
}

void BroadcastPreWeightsIO::readHeader() {
   BufferUtils::WeightHeader header;
   getFileStream()->read(&header, mHeaderSize);
   FatalIf(
         checkHeader(header) != PV_SUCCESS,
         "BroadcastPreWeightsIO::readHeader() in \"%s\", frame %d\n",
         getFileStream()->getFileName().c_str(), getFrameNumber());
   mHeader.baseHeader.timestamp = header.baseHeader.timestamp;
   mHeader.minVal = header.minVal;
   mHeader.maxVal = header.maxVal;
   setFrameNumber(getFrameNumber());
}

void BroadcastPreWeightsIO::readHeader(int frameNumber) {
   setFrameNumber(frameNumber);
   readHeader();
}

void BroadcastPreWeightsIO::readRegion(
      WeightData &weightData,
      int xStart,
      int yStart,
      int fStart,
      int fPreStart,
      int arborIndexStart) {
   if (!mFileStream) {
      return;
   }

   if (weightData.getNumArbors() == 0) {
      return;
   }

   int nxpLocal   = weightData.getPatchSizeX();
   int nypLocal   = weightData.getPatchSizeY();
   int nfpLocal   = weightData.getPatchSizeF();
   int lineSize   = nxpLocal * nfpLocal;
   int nfPreLocal = weightData.getNumDataPatchesF();
   int numArbors = weightData.getNumArbors();
   /*
    * Check dimensions:
    *     xStart + nxpLocal should be <= mHeader.nxp
    *     yStart + nypLocal should be <= mHeader.nyp
    *     fStart + nfpLocal should be <= mHeader.nfp
    *     mHeader.baseHeader.nx{,Extended} and weightData.getPatchSizeX() should each be 1
    *     mHeader.baseHeader.ny{,Extended} and weightData.getPatchSizeY() should each be 1
    *     fPreStart + nfPreLocal should be <= mHeader.baseHeader.nf
    *     arborIndexStart + numArbors should be <= mHeader.nbands
    */
   vector<uint8_t> compressedValuesBuffer(getCompressedFlag() ? lineSize : 0);
   for (int a = 0; a < numArbors; ++a) {
      for (int p = 0; p < nfPreLocal; ++p) {
         for (int y = 0; y < nypLocal; ++y) {
            float *localPointer = &weightData.getDataFromDataIndex(a, p)[y * lineSize];
            // Set file location
            int lineStartInFilePatch = kIndex(
                  xStart, yStart + y, fStart, mPatchSizeX, mPatchSizeY, mPatchSizeF);
            long lineStartOffset = static_cast<long int>(lineStartInFilePatch) * mDataSize;
            long frameStart      = calcFilePositionFromFrameNumber(getFrameNumber()) + mHeaderSize;
            long arborStart      = frameStart + (arborIndexStart + a) * calcArborSizeBytes();
            long patchStart      = arborStart + (p + fPreStart) * calcPatchSizeBytes();
            long patchDataStart  = patchStart + mPatchHeaderSize;
            long lineStart       = patchDataStart + lineStartOffset;
            getFileStream()->setInPos(lineStart, std::ios_base::beg);
            if (getCompressedFlag()) {
               getFileStream()->read(compressedValuesBuffer.data(), lineSize);
               for (int k = 0; k < lineSize; ++k) {
                  float compressedVal   = static_cast<float>(compressedValuesBuffer[k]) / 255.0f;
                  float uncompressedVal = compressedVal *(mHeader.maxVal - mHeader.minVal);
                  uncompressedVal += mHeader.minVal;
                  localPointer[k] = uncompressedVal;
               }
            }
            else {
               getFileStream()->read(localPointer, lineSize * mDataSize);
            }
         }
      }
   }
   // Return filepointer to beginning of frame
   setFrameNumber(getFrameNumber());
}

void BroadcastPreWeightsIO::writeHeader() {
   getFileStream()->write(&mHeader, mHeaderSize);
   setFrameNumber(getFrameNumber());
}

void BroadcastPreWeightsIO::writeHeader(int frameNumber) {
   setFrameNumber(frameNumber);
   writeHeader();
}

void BroadcastPreWeightsIO::writeRegion(
      WeightData const &weightData,
      int xStart,
      int yStart,
      int fStart,
      int fPreStart,
      int arborIndexStart) {
   if (!mFileStream) {
      return;
   }

   if (weightData.getNumArbors() == 0) {
      return;
   }

   int nxpLocal   = weightData.getPatchSizeX();
   int nypLocal   = weightData.getPatchSizeY();
   int nfpLocal   = weightData.getPatchSizeF();
   int lineSize   = nxpLocal * nfpLocal;
   int nfPreLocal = weightData.getNumDataPatchesF();
   int numArbors = weightData.getNumArbors();
   /*
    * Check dimensions:
    *     xStart + nxpLocal should be <= mHeader.nxp
    *     yStart + nypLocal should be <= mHeader.nyp
    *     fStart + nfpLocal should be <= mHeader.nfp
    *     mHeader.baseHeader.nx{,Extended} and weightData.getPatchSizeX() should each be 1
    *     mHeader.baseHeader.ny{,Extended} and weightData.getPatchSizeY() should each be 1
    *     fPreStart + nfPreLocal should be <= mHeader.baseHeader.nf
    *     arborIndexStart + numArbors should be <= mHeader.nbands
    */
   vector<uint8_t> compressedValuesBuffer(getCompressedFlag() ? lineSize : 0);
   for (int a = 0; a < numArbors; ++a) {
      for (int p = 0; p < nfPreLocal; ++p) {
         long frameStart      = calcFilePositionFromFrameNumber(getFrameNumber()) + mHeaderSize;
         long arborStart      = frameStart + (arborIndexStart + a) * calcArborSizeBytes();
         long patchStart      = arborStart + (p + fPreStart) * calcPatchSizeBytes();
         getFileStream()->setOutPos(patchStart, std::ios_base::beg);
         Patch patchHeader;
         patchHeader.nx = static_cast<std::uint16_t>(mHeader.nxp);
         patchHeader.ny = static_cast<std::uint16_t>(mHeader.nyp);
         patchHeader.offset = static_cast<std::uint32_t>(0);
         getFileStream()->write(&patchHeader, sizeof(patchHeader));
         long patchDataStart  = patchStart + mPatchHeaderSize;
         for (int y = 0; y < nypLocal; ++y) {
            float const *localPointer = &weightData.getDataFromDataIndex(a, p)[y * lineSize];
            // Set file location
            int lineStartInFilePatch = kIndex(
                  xStart, yStart + y, fStart, mPatchSizeX, mPatchSizeY, mPatchSizeF);
            long lineStartOffset = static_cast<long int>(lineStartInFilePatch) * mDataSize;
            long lineStart       = patchDataStart + lineStartOffset;
            getFileStream()->setOutPos(lineStart, std::ios_base::beg);
            if (getCompressedFlag()) {
               for (int k = 0; k < lineSize; ++k) {
                  float uncompressedVal = localPointer[k];
                  float compressedVal = (uncompressedVal - mHeader.minVal);
                  compressedVal /= mHeader.maxVal - mHeader.minVal;
                  compressedValuesBuffer[k] = static_cast<uint8_t>(compressedVal * 255.0f);
               }
               getFileStream()->write(compressedValuesBuffer.data(), lineSize);
            }
            else {
               getFileStream()->write(localPointer, lineSize * mDataSize);
            }
         }
      }
   }
   // Return filepointer to beginning of frame
   setFrameNumber(getFrameNumber());
}

void BroadcastPreWeightsIO::open() {
   mFileStream->open();
   initializeNumFrames();
}

void BroadcastPreWeightsIO::close() { mFileStream->close(); }

void BroadcastPreWeightsIO::setFrameNumber(int frameNumber) {
   pvAssert(mFileStream);
   mFrameNumber = frameNumber;
   long filePos = calcFilePositionFromFrameNumber(frameNumber);
   mFileStream->setInPos(filePos, std::ios_base::beg);
   if (mFileStream->writeable()) {
      mFileStream->setOutPos(filePos, std::ios_base::beg);
   }
}

long BroadcastPreWeightsIO::calcArborSizeBytes() const {
   long patchSizeBytes = calcPatchSizeBytes();
   long numPatches     = getNfPre();
   long sizeBytes      = numPatches * patchSizeBytes;
   return sizeBytes;
}

long BroadcastPreWeightsIO::calcFrameSizeBytes() const {
   long sizeBytes = mHeaderSize + static_cast<long>(mNumArbors) * calcArborSizeBytes();
   return sizeBytes;
}

long BroadcastPreWeightsIO::calcPatchSizeBytes() const {
   return mDataSize * getPatchSizeOverall() + mPatchHeaderSize;
}

std::array<std::vector<int>, 2> BroadcastPreWeightsIO::calcPatchStartsAndStops(
      int nExtendedPre,
      int nRestrictedPre,
      int nPreRef,
      int nPostRef,
      int patchSize) {
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
            k,
            nRestrictedPre,
            beginMargin,
            endMargin,
            nPost,
            0,
            0,
            patchSize,
            &dim,
            &start,
            &dummy1,
            &dummy2,
            &dummy3);
      result[0].at(k) = start;
      result[1].at(k) = start + dim;
   }
   return result;
}

int BroadcastPreWeightsIO::checkHeader(BufferUtils::WeightHeader const &header) const {
   int status = PV_SUCCESS;
   BufferUtils::ActivityHeader const &expected = mHeader.baseHeader;
   BufferUtils::ActivityHeader const &observed = header.baseHeader;
   status = checkHeaderField(expected.headerSize, observed.headerSize, "headerSize", status);
   status = checkHeaderField(expected.numParams, observed.numParams, "numParams", status);
   status = checkHeaderField(expected.fileType, observed.fileType, "fileType", status);
   status = checkHeaderField(expected.nx, observed.nx, "nx", status);
   status = checkHeaderField(expected.ny, observed.ny, "ny", status);
   status = checkHeaderField(expected.nf, observed.nf, "nf", status);
   status = checkHeaderField(expected.numRecords, observed.numRecords, "numRecords", status);
   status = checkHeaderField(expected.recordSize, observed.recordSize, "recordSize", status);
   status = checkHeaderField(expected.dataSize, observed.dataSize, "dataSize", status);
   status = checkHeaderField(expected.dataType, observed.dataType, "dataType", status);
   status = checkHeaderField(expected.nxProcs, observed.nxProcs, "nxProcs", status);
   status = checkHeaderField(expected.nyProcs, observed.nyProcs, "nyProcs", status);
   status = checkHeaderField(expected.nxExtended, observed.nxExtended, "nxExtended", status);
   status = checkHeaderField(expected.nyExtended, observed.nyExtended, "nyExtended", status);
   status = checkHeaderField(expected.kx0, observed.kx0, "kx0", status);
   status = checkHeaderField(expected.ky0, observed.ky0, "ky0", status);
   status = checkHeaderField(expected.nBatch, observed.nBatch, "nBatch", status);
   status = checkHeaderField(expected.nBands, observed.nBands, "nBands", status);
   // timestamp does not have to match
   status = checkHeaderField(mHeader.nxp, header.nxp, "nxp", status);
   status = checkHeaderField(mHeader.nyp, header.nyp, "nyp", status);
   status = checkHeaderField(mHeader.nfp, header.nfp, "nfp", status);
   // minVal, maxVal do not have to match
   status = checkHeaderField(mHeader.numPatches, header.numPatches, "numPatches", status);
   return status;
}

int BroadcastPreWeightsIO::checkHeaderField(
      int expected, int observed, std::string const &fieldLabel, int oldStatus) const {
   if (expected != observed) {
      ErrorLog().printf(
            "BroadcastPreWeights file \"%s\" header field %s should be %d, but it is %d\n",
            mFileStream->getFileName().c_str(), fieldLabel.c_str(), expected, observed);
      return PV_FAILURE;
   }
   return oldStatus;
}

int BroadcastPreWeightsIO::checkHeaderField(
      double expected, double observed, std::string const &fieldLabel, int oldStatus) const {
   if (expected != observed) {
      ErrorLog().printf(
            "BroadcastPreWeights file \"%s\" header field %s should be %f, but it is %f\n",
            mFileStream->getFileName().c_str(), fieldLabel.c_str(), expected, observed);
      return PV_FAILURE;
   }
   return oldStatus;
}

void BroadcastPreWeightsIO::initializeFrameSize() {
   mFrameSize = mHeaderSize + static_cast<long>(mNumArbors) * calcArborSizeBytes();
}

void BroadcastPreWeightsIO::initializeHeader() {
   BufferUtils::ActivityHeader &baseHeader = mHeader.baseHeader;
   baseHeader.headerSize = NUM_WGT_PARAMS * static_cast<int>(sizeof(float));
   baseHeader.numParams  = NUM_WGT_PARAMS;
   baseHeader.fileType   = PVP_WGT_FILE_TYPE;
   baseHeader.nx         = 1;
   baseHeader.ny         = 1;
   baseHeader.nf         = getNfPre();
   baseHeader.numRecords = getNumArbors();
   baseHeader.recordSize = 0;
   if (getCompressedFlag()) {
      baseHeader.dataSize = static_cast<int>(sizeof(uint8_t));
      baseHeader.dataType = BufferUtils::BYTE;
   }
   else {
      baseHeader.dataSize = static_cast<int>(sizeof(float));
      baseHeader.dataType = BufferUtils::FLOAT;
   }
   baseHeader.nxProcs    = 1;
   baseHeader.nyProcs    = 1;
   baseHeader.nxExtended = 1;
   baseHeader.nyExtended = 1;
   baseHeader.kx0        = 0;
   baseHeader.ky0        = 0;
   baseHeader.nBatch     = 1;
   baseHeader.nBands     = getNumArbors();
   baseHeader.timestamp  = 0.0;

   mHeader.nxp        = getPatchSizeX();
   mHeader.nyp        = getPatchSizeY();
   mHeader.nfp        = getPatchSizeF();
   mHeader.minVal     = std::numeric_limits<float>::lowest();
   mHeader.maxVal     = std::numeric_limits<float>::max();
   mHeader.numPatches = getNfPre();

   mHeaderWrittenFlag = false;
}

void BroadcastPreWeightsIO::initializeNumFrames() {
   if (!getFileStream()) {
      return;
   }

   long curPos = getFileStream()->getInPos();
   getFileStream()->setInPos(0L, std::ios_base::end);
   long eofPosition = getFileStream()->getInPos();
   mNumFrames       = calcFrameNumberFromFilePosition(eofPosition);
   getFileStream()->setInPos(curPos, std::ios_base::beg);
}

} // namespace PV
