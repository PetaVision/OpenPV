#include "SharedWeightsIO.hpp"

#include "utils/PVLog.hpp"

namespace PV {

SharedWeightsIO::SharedWeightsIO(std::shared_ptr<FileStream> fileStream) :
      mFileStream(fileStream) {
   if (!getFileStream()) { return; }
   FatalIf(
         !getFileStream()->readable(),
         "FileStream \"%s\" is not readable and can't be used in a SharedWeightsIO object.\n",
         getFileStream()->getFileName());

   initializeFrameStarts();

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

long SharedWeightsIO::calcFilePositionFromFrameNumber(int frameNumber) const {
   if (frameNumber >= 0 and frameNumber < static_cast<int>(mFrameStarts.size())) {
      return mFrameStarts[frameNumber];
   }
   else {
      return -1L;
   }
}

int SharedWeightsIO::calcFrameNumberFromFilePosition(long filePosition) const {
   pvAssert(getFrameNumber() >= 0 and getFrameNumber() <= getNumFrames());
   pvAssert(mFrameStarts.size() == static_cast<std::vector<long>::size_type>(getNumFrames() + 1));
   pvAssert(mFrameStarts[0] == 80L);
#ifndef NDEBUG
   // If in debug mode, verify that the entries of mFrameStarts are strictly increasing,
   // and the first element is the size of the header in bytes.
   for (int k = 0; k < getNumFrames(); ++k) {
      pvAssert(mFrameStarts[k] < mFrameStarts[k+1]);
   }
#endif // NDEBUG
   auto p = mFrameStarts.begin();
   while(p != mFrameStarts.end()) {
      if (*p >= filePosition) { break; }
      ++p;
   }
   if (p == mFrameStarts.end()) { return -1; }

   int frameNumber = static_cast<int>(p - mFrameStarts.begin());
   return frameNumber;
}

void SharedWeightsIO::read(Weights &weights) {
   double dummyTimestamp;
   read(weights, dummyTimestamp);
}

void SharedWeightsIO::read(Weights &weights, double &timestamp) {
   BufferUtils::WeightHeader header;
   uint32_t headerSize = 0U;
   getFileStream()->read(&headerSize, 4L);
   FatalIf(headerSize != static_cast<uint32_t>(mHeaderSize),
         "SharedWeightsIO::read() \"%s\" frame %d has headerSize %" PRIu32
         " when it should be %" PRIu32 ".\n",
         getFileStream()->getFileName(),
         getFrameNumber(),
         headerSize,
         static_cast<uint32_t>(mHeaderSize));
   getFileStream()->setInPos(-4L, std::ios_base::cur);
   getFileStream()->read(&header, static_cast<long>(mHeaderSize));
   checkHeaderValues(weights, header);

   timestamp = header.baseHeader.timestamp;
   long arborSize = calcArborSize(header);
   std::vector<unsigned char> readBuffer(arborSize);
   int numArbors   = header.baseHeader.numRecords;
   bool compressed = header.baseHeader.dataSize == BufferUtils::BYTE;
   for (int arbor = 0; arbor < numArbors; ++arbor) {
      mFileStream->read(readBuffer.data(), arborSize);
      loadWeightsFromBuffer(readBuffer, arbor, header.minVal, header.maxVal, compressed);
   }
   setFrameNumber(getFrameNumber() + 1);
}

void SharedWeightsIO::read(Weights &weights, double &timestamp, int frameNumber) {
   setFrameNumber(frameNumber);
   read(weights, timestamp);
}

void SharedWeightsIO::write(Weights &weights, bool compressed, double timestamp) {
   float minWeight = weights.calcMinWeight();
   float maxWeight = weights.calcMaxWeight();
   auto header = writeHeader(weights, compressed, timestamp, minWeight, maxWeight);
   long arborSize = calcArborSize(header);
   std::vector<unsigned char> write(arborSize);
   int numArbors   = header.baseHeader.numRecords;
   for (int arbor = 0; arbor < numArbors; ++arbor) {
      storeWeightsInBuffer(writeBuffer, arbor, minWeight, maxWeight, compress);
      mFileStream->write(readBuffer.data(), arborSize);
   }
   if (mFrameNumber == mNumFrames) {
      mFrameStarts.push_back(getFileStream()->getOutPos());
      ++mNumFrames;
      setFrameNumber(mNumFrames);
   }
   else {
      setFrameNumber(mFrameNumber + 1);
   }
}

void SharedWeightsIO::write(Weights &weights, bool compressed, double timestamp, int frameNumber) {
   setFrameNumber(frameNumber);
   write(weights, compressed, timestamp);
}

void SharedWeightsIO::setFrameNumber(int frame) {
   if (!mFileStream) { return; }
   mFrameNumber = frame;
   long filePos = calcFilePositionFromFrameNumber(frame);
   mFileStream->setInPos(filePos, std::ios_base::beg);
   if (mFileStream->writeable()) { mFileStream->setOutPos(filePos, std::ios_base::beg); }
}

long SharedWeightsIO::calcArborSize(BufferUtils::WeightHeader const &header) const {
   long patchDataSize    = static_cast<long>(nxp * nyp * nfp * header.baseHeader.dataSize);
   auto patchHeaderSize  = std::sizeof(uint16_t) + std::sizeof(uint16_t) + std::sizeof(uint32_t);
   long patchSizeInBytes = static_cast<long>(patchHeaderSize) + patchDataSize;
   long arborSizeInBytes = patchSizeInBytes * static_cast<long>(header.numPatches);
   return arborSizeInBytes;
}

void SharedWeightsIO::checkHeaderValues(
      Weights const &weights, BufferUtils::WeightHeader const &header) const {
   auto dataType = static_cast<BufferUtils::HeaderDataType>(header.baseHeader.dataType);
   FatalIf(
         dataType != BufferUtils::BYTE and dataType != BufferUtils::FLOAT,
         "SharedWeights file \"%s\" frame %d has dataType %d. "
         "Only BYTE(%d) and FLOAT(%d) are supported.\n",
         getFileStream()->getFileName(), getFrameNumber(), static_cast<int>(dataType),
         static_cast<int>(BufferUtils::BYTE), static_cast<int>(BufferUtils::FLOAT));
   FatalIf(header.nxp != weights.getPatchSizeX(),
         "SharedWeights file \"%s\" frame %d has nxp %d, "
         "which is incompatible with target weight's nxp = %d\n"
         getFileStream()->getFileName(), getFrameNumber(), header.nxp,
         weights.getPatchSizeX());
   FatalIf(header.nyp != weights.getPatchSizeY(),
         "SharedWeights file \"%s\" frame %d has nyp %d, "
         "which is incompatible with target weight's nyp = %d\n"
         getFileStream()->getFileName(), getFrameNumber(), header.nyp,
         weights.getPatchSizeY());
   FatalIf(header.nfp != weights.getPatchSizeF(),
         "SharedWeights file \"%s\" frame %d has nfp %d, "
         "which is incompatible with target weight's nfp = %d\n"
         getFileStream()->getFileName(), getFrameNumber(), header.nfp,
         weights.getPatchSizeF());
   FatalIf(header.numPatches != weights.getNumDataPatches(),
         "SharedWeights file \"%s\" frame %d has numPatches %d, "
         "which is incompatible with target weight's NumDataPatches = %d\n"
         getFileStream()->getFileName(), getFrameNumber(), header.numPatches,
         weights.getNumDataPatches());
   FatalIf(header.numRecords != weights.getNumArbors(),
         "SharedWeights file \"%s\" frame %d has %d arbors, "
         "which is incompatible with target weight's NumArbors = %d\n"
         getFileStream()->getFileName(), getFrameNumber(), header.numRecords,
         weights.getNumArbors());
   FatalIf(header.nBands != weights.numRecords,
         "SharedWeights file \"%s\" frame %d has inconsistent numbers of arbors: "
         "numRecords = %d but nBands = %d.\n",
         getFileStream()->getFileName(), getFrameNumber(),
         header.baseHeader.numRecords, header.baseHeader.nBands);
}

void WeightsFileIO::compressPatch(
      unsigned char *writeBuffer,
      float const *sourceWeights,
      int count,
      float minValue,
      float maxValue) {
   for (int k = 0; k < count; k++) {
      float compressedWeight = (sourceWeights[k] - minValue) / (maxValue - minValue);
      float scaledWeight     = std::floor(255.0f * compressedWeight);
      writeBuffer[k]         = static_cast<unsigned char>(scaledWeight);
   }
}

void SharedWeightsIO::decompressPatch(
      unsigned char const *readBuffer,
      float *destWeights,
      int count,
      float minValue,
      float maxValue) {
   for (int k = 0; k < count; k++) {
      float compressedWeight = static_cast<float>(readBuffer[k]) / 255.0f;
      destWeights[k]         = compressedWeight * (maxValue - minValue) + minValue;
   }
}

void SharedWeightsIO::initializeFrameStarts() {
   // Should only be called by constructor, after nonroot process have returned
   pvAssert(getFileStream());

   getFileStream()->setInPos(0L, std::ios_base::end);
   long eofPosition = getFileStream()->getInPos();
   FatalIf(
         eofPosition < mHeaderSize,
         "SparseLayerIO \"%s\" is too shore (%ld bytes) to contain a weight header.\n",
         getFileStream()->getFileName(), eofPosition);
   long curPosition = 0L;
   getFileStream()->setInPos(curPosition, std::ios_base::beg);
   while (curPosition < eofPosition) {
      mFrameStarts.push_back(curPosition);

      // Make sure there's enough data left in the file for timestamp + numActive
      FatalIf(
            eofPosition - curPosition < static_cast<long>(mHeaderSize),
            "SharedWeightsIO \"%s\" has %ld bytes left over after %zu pvp frames.\n",
            getFileStream()->getFileName(),
            eofPosition - curPosition,
            mFrameStarts.size());

      // Read timestamp and numActive
      WeightHeader header;
      getFileStream()->read(&header, static_cast<long>(mHeaderSize));

      // Make sure there's enough data left in the file for the weight data.
      long arborSize = calcArborSize(header);
      long frameDataSize = arborSize * static_cast<long>(header.baseHeader.numRecords);
      FatalIf(
            eofPosition - curPosition < frameDataSize,
            "SparseLayerIO \"%s\" has numActive=%d in frame %zu, and therefore needs "
            "%d bytes to hold the values, but there are only %ld bytes left in the file.\n",
            getFileStream()->getFileName(),
            mFrameStarts.size(),
            numActive * static_cast<int>(mSparseValueEntrySize),
            eofPosition - curPosition);

      long newPosition = getFileStream()->getInPos();
      pvAssert(newPosition == curPosition + 104L);
      curPosition = newPosition + frameDataSize;
      getFileStream()->setInPos(curPosition, std::ios_base::beg);
   }
   pvAssert(curPosition == eofPosition);
   mNumFrames = static_cast<int>(mNumEntries.size());
   mFrameStarts.push_back(eofPosition);

   pvAssert(mFrameStarts.size() == static_cast<std::vector<long>::size_type>(mNumFrames + 1));
}

void SharedWeightsIO::loadWeightsFromBuffer(
      Weights &weights,
      std::vector<unsigned char> const &readBuffer,
      int arbor,
      float minValue,
      float maxValue,
      bool compressed) {
   int const nxp           = weights.getPatchSizeX();
   int const nyp           = weights.getPatchSizeY();
   int const nfp           = weights.getPatchSizeF();
   int const valuesInPatch = nxp * nyp * nfp;
   int const numPatches    = weights.getNumDataPatches();

   auto const patchSizePvpFormat     = BufferUtils::weightPatchSize(valuesInPatch, compressed);
   std::size_t const patchHeaderSize = sizeof(unsigned int) + 2UL * sizeof(unsigned short);
   if (compressed) {
      for (int k = 0; k < numPatches; k++) {
         std::size_t const offsetInFile     = patchSizePvpFormat * (std::size_t)k;
         unsigned char const *patchFromFile = &readBuffer[offsetInFile + patchHeaderSize];
         float *weightsInPatch              = weights.getDataFromDataIndex(arbor, k);
         decompressPatch(patchFromFile, weightsInPatch, valuesInPatch, minValue, maxValue);
      }
   }
   else {
      for (int k = 0; k < numPatches; k++) {
         std::size_t const offsetInFile     = patchSizePvpFormat * (std::size_t)k;
         unsigned char const *patchFromFile = &readBuffer[offsetInFile + patchHeaderSize];

         float *weightsInPatch   = weights.getDataFromDataIndex(arbor, k);
         std::size_t sizeInBytes = static_cast<std::size_t>(valuesInPatch) * sizeof(float);
         memcpy(weightsInPatch, patchFromFile, sizeInBytes);
      }
   }
}

void SharedWeightsIO::storeWeightsInBuffer(
      Weights const &weights,
      std::vector<unsigned char> &readBuffer,
      int arbor,
      float minValue,
      float maxValue,
      bool compressed) {
   int const nxp           = weights.getPatchSizeX();
   int const nyp           = weights.getPatchSizeY();
   int const nfp           = weights.getPatchSizeF();
   int const valuesInPatch = nxp * nyp * nfp;
   int const numPatches    = weights.getNumDataPatches();

   auto const patchSizePvpFormat     = BufferUtils::weightPatchSize(valuesInPatch, compressed);
   std::size_t const patchHeaderSize = sizeof(uint32_t) + 2UL * sizeof(uint16_t);
   unsigned char patchHeader[patchHeaderSize];
   uint16_t shortDim[2] = {static_cast<uint16_t>(nxp), static_cast<uint16_t>(nyp)};
   memcpy(patchHeader, &shortDim, 2UL * sizeof(uint16_t));

   // always zero offset for shared weights
   memset(&patchHeader[2UL * sizeof(uint16_t)], 0, sizeof(uint32_t));
   if (compressed) {
      for (int k = 0; k < numDataPatches; k++) {
         std::size_t const offsetInFile = patchSizePvpFormat * (std::size_t)k;
         unsigned char *patchFromFile   = &dataFromFile[offsetInFile] + patchHeaderSize;

         float const *weightsInPatch = weights.getDataFromDataIndex(arbor, k);
         compressPatch(patchFromFile, weightsInPatch, valuesInPatch, minValue, maxValue);
      }
   }
   else {
      for (int k = 0; k < numDataPatches; k++) {
         std::size_t const offsetInFile = patchSizePvpFormat * (std::size_t)k;
         unsigned char *patchFromFile   = &dataFromFile[offsetInFile] + patchHeaderSize;

         float const *weightsInPatch = weights.getDataFromDataIndex(arbor, k);
         memcpy(patchFromFile, weightsInPatch, (std::size_t)(valuesInPatch) * sizeof(float));
      }
   }
}

BufferUtils::WeightHeader SharedWeightsIO::writeHeader(
      Weights &weights,
      bool compressed,
      double timestamp,
      float minWeight,
      float maxWeight) {
   BufferUtils::WeightHeader header = BufferUtils::buildSharedWeightHeader(
         weights.getPatchSizeX(),
         weights.getPatchSizeY(),
         weights.getPatchSizeF(),
         weights.getNumArbors(),
         weights.getNumDataPatchesX(),
         weights.getNumDataPatchesY(),
         weights.getNumDataPatchesF(),
         timestamp,
         compressed,
         minWeight,
         maxWeight);
   getFileStream()->write(&header, mHeaderSize);
   return header;
}

} // namespace PV
