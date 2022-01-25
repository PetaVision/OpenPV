#include "SparseLayerIO.hpp"

#include "structures/Buffer.hpp"
#include "structures/SparseList.hpp"
#include "utils/PVAssert.hpp"

namespace PV {

SparseLayerIO::SparseLayerIO(
      std::shared_ptr<FileStream> fileStream,
      int width,
      int height,
      int numFeatures) :
      mFileStream(fileStream), mWidth(width), mHeight(height), mNumFeatures(numFeatures) {
   FatalIf(
         fileStream and !fileStream->readable(),
         "FileStream \"%s\" is not readable and can't be used in a SparseLayerIO object.\n",
         fileStream->getFileName());

   if (!getFileStream()) { return; }

   getFileStream()->setInPos(0L, std::ios_base::end);
   long eofPosition = getFileStream()->getInPos();
   if (getFileStream()->writeable()) {
      if (eofPosition == 0L) { writeHeader(); }
   }

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

long SparseLayerIO::calcFilePositionFromFrameNumber(int frameNumber) const {
   if (frameNumber >= 0 and frameNumber < static_cast<int>(mFrameStarts.size())) {
      return mFrameStarts[frameNumber];
   }
   else {
      return -1L;
   }
}

int SparseLayerIO::calcFrameNumberFromFilePosition(long filePosition) const {
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

void SparseLayerIO::read(SparseList<float> &sparseList) {
   double dummyTimestamp;
   read(sparseList, dummyTimestamp);
}

void SparseLayerIO::read(SparseList<float> &sparseList, double &timestamp) {
   if (!getFileStream()) { return; }
   getFileStream()->read(&timestamp, 8L);
   int numValues;
   getFileStream()->read(&numValues, 4L);
   std::vector<SparseList<float>::Entry> entryVector(numValues);
   long numBytes = static_cast<long>(numValues) * mSparseValueEntrySize;
   getFileStream()->read(entryVector.data(), numBytes);
   sparseList.set(entryVector);
   setFrameNumber(getFrameNumber() + 1);
}

void SparseLayerIO::read(SparseList<float> &sparseList, double &timestamp, int frameNumber) {
   setFrameNumber(frameNumber);
   read(sparseList, timestamp);
}

void SparseLayerIO::write(SparseList<float> const &sparseList, double timestamp) {
   if (!getFileStream()) { return; }
   FatalIf(mFrameNumber != mNumFrames,
         "Writing to frame %d of \"%s\", which has %d frames, "
         "but we haven't implemented writing to the middle of a sparse frame yet.\n",
         mFrameNumber, getFileStream()->getFileName().c_str(), mNumFrames);

   getFileStream()->write(&timestamp, 8L);
   auto listContents = sparseList.getContents();
   int numValues = static_cast<int>(listContents.size());
   long numBytes = static_cast<long>(numValues) * mSparseValueEntrySize;
   getFileStream()->write(&numValues, 4L);
   getFileStream()->write(listContents.data(), numBytes);
   if (mFrameNumber == mNumFrames) {
      mFrameStarts.push_back(getFileStream()->getOutPos());
      mNumEntries.push_back(numValues);
      ++mNumFrames;
      setFrameNumber(mNumFrames);
      setHeaderNBands();
   }
   else {
      // Hasn't been implemented yet
      setFrameNumber(mFrameNumber + 1);
   }
}

void SparseLayerIO::write(SparseList<float> const &sparseList, double timestamp, int frameNumber) {
   setFrameNumber(frameNumber);
   write(sparseList, timestamp);
}

void SparseLayerIO::open() {
   if (mFileStream) {
      mFileStream->open();
      initializeFrameStarts();
   }
}

void SparseLayerIO::close() {
   if (mFileStream) { mFileStream->close(); }
}

void SparseLayerIO::setFrameNumber(int frame) {
   if (!mFileStream) { return; }
   mFrameNumber = frame;
   long filePos = calcFilePositionFromFrameNumber(frame);
   mFileStream->setInPos(filePos, std::ios_base::beg);
   if (mFileStream->writeable()) { mFileStream->setOutPos(filePos, std::ios_base::beg); }
}

void SparseLayerIO::initializeFrameStarts() {
   // Should only be called by root process, either by constructor or on (re-)opening the file
   pvAssert(getFileStream());

   getFileStream()->setInPos(0L, std::ios_base::end);
   long eofPosition = getFileStream()->getInPos();
   FatalIf(
         eofPosition < mHeaderSize,
         "SparseLayerIO \"%s\" is too short (%ld bytes) to contain an activity header.\n",
         getFileStream()->getFileName(), eofPosition);
   long curPosition = mHeaderSize;
   getFileStream()->setInPos(curPosition, std::ios_base::beg);
   mFrameStarts.clear();
   mNumEntries.clear();
   while (curPosition < eofPosition) {
      mFrameStarts.push_back(curPosition);

      // Make sure there's enough data left in the file for timestamp + numActive
      FatalIf(
            eofPosition - curPosition < 12L,
            "SparseLayerIO \"%s\" has %ld bytes left over after %zu pvp frames.\n",
            getFileStream()->getFileName(),
            eofPosition - curPosition,
            mFrameStarts.size());

      // Read timestamp and numActive
      double timestamp;
      getFileStream()->read(&timestamp, 8L);
      int numActive;
      getFileStream()->read(&numActive, 4L);

      // Make sure there's enough data left in the file for the sparse-value entries
      long needed = static_cast<long>(numActive) * mSparseValueEntrySize;
      FatalIf(
            eofPosition - curPosition < needed,
            "SparseLayerIO \"%s\" has numActive=%d in frame %zu, and therefore needs "
            "%ld bytes to hold the values, but there are only %ld bytes left in the file.\n",
            getFileStream()->getFileName().c_str(),
            numActive,
            needed,
            eofPosition - curPosition);

      mNumEntries.push_back(numActive);
      long newPosition = getFileStream()->getInPos();
      pvAssert(newPosition == curPosition + 12L);
      curPosition = newPosition + mSparseValueEntrySize * static_cast<long>(numActive);
      getFileStream()->setInPos(curPosition, std::ios_base::beg);
   }
   pvAssert(curPosition == eofPosition);
   mFrameStarts.push_back(eofPosition);

   mNumFrames = static_cast<int>(mNumEntries.size());

   pvAssert(mFrameStarts.size() == static_cast<std::vector<long>::size_type>(mNumFrames + 1));
}

void SparseLayerIO::setHeaderNBands() {
   pvAssert(getFileStream());
   long currentPosition = getFileStream()->getOutPos();
   getFileStream()->setOutPos(0L, std::ios_base::end);
   long fileLength = getFileStream()->getOutPos();
   int nBandsTotal = calcFrameNumberFromFilePosition(fileLength);
   long nBandsPosition = 68L;
   getFileStream()->setOutPos(nBandsPosition, std::ios_base::beg);
   getFileStream()->write(&nBandsTotal, static_cast<long>(sizeof(nBandsTotal)));
   getFileStream()->setOutPos(currentPosition, std::ios_base::beg);
}

void SparseLayerIO::writeHeader() {
   if (!getFileStream()) { return; }
   FatalIf(
         !getFileStream()->writeable(),
         "writeHeader() called but \"%s\" is not writeable.\n",
         getFileStream()->getFileName());

   BufferUtils::ActivityHeader header;
   header.headerSize = static_cast<int>(mHeaderSize);
   header.numParams  = 20;
   header.fileType   = PVP_ACT_SPARSEVALUES_FILE_TYPE;
   header.nx         = mWidth;
   header.ny         = mHeight;
   header.nf         = mNumFeatures;
   header.numRecords = 1;
   header.recordSize = 0; // sparse files have no fixed record size
   header.dataSize   = static_cast<int>(mSparseValueEntrySize);
   header.dataType   = BufferUtils::returnDataType<float>();
   header.nxProcs    = 1;
   header.nyProcs    = 1;
   header.nxExtended = mWidth;
   header.nyExtended = mHeight;
   header.kx0        = 0;
   header.ky0        = 0;
   header.nBatch     = 1;
   header.nBands     = 0;
   header.timestamp  = 0.0;

   getFileStream()->write(&header, static_cast<long>(sizeof(header)));
}

} // namespace PV
