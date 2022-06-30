#include "LayerIO.hpp"

#include "structures/Buffer.hpp"
#include "utils/BufferUtilsMPI.hpp" // gather, scatter
#include "utils/PVAssert.hpp"

namespace PV {

LayerIO::LayerIO(
      std::shared_ptr<FileStream> fileStream,
      int width,
      int height,
      int numFeatures) :
      mFileStream(fileStream), mWidth(width), mHeight(height), mNumFeatures(numFeatures) {
   FatalIf(
         fileStream and !fileStream->readable(),
         "FileStream \"%s\" is not readable and can't be used in a LayerIO object.\n",
         fileStream->getFileName().c_str());

   if (!getFileStream()) { return; }

   initializeNumFrames();

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

long LayerIO::calcFilePositionFromFrameNumber(int frameNumber) const {
   long const frameHeaderSize   = 8L;
   long const valuesPerPVPFrame = calcValuesPerPVPFrame();
   long const frameSizeInBytes  = frameHeaderSize + mDataSize * valuesPerPVPFrame;

   long const filePosition =
         mHeaderSize + frameSizeInBytes * static_cast<long>(frameNumber);
   return filePosition;
}

int LayerIO::calcFrameNumberFromFilePosition(long filePosition) const {
   long const frameHeaderSize = 8L;

   long const valuesPerPVPFrame = calcValuesPerPVPFrame();
   long const frameSizeInBytes  = frameHeaderSize + mDataSize * valuesPerPVPFrame;

   long const frameNumber = (filePosition - mHeaderSize) / frameSizeInBytes;
   return static_cast<int>(frameNumber);
}

void LayerIO::read(Buffer<float> &buffer) {
   double dummyTimestamp;
   read(buffer, dummyTimestamp);
}

void LayerIO::read(Buffer<float> &buffer, double &timestamp) {
   if (!mFileStream) { return; }

   if (buffer.getTotalElements() == 0) {
      buffer.resize(mWidth, mHeight, mNumFeatures);
   }

   long numBytes = checkBufferDimensions(buffer);
   mFileStream->read(&timestamp, 8L);
   mFileStream->read(buffer.asVector().data(), numBytes);

   setFrameNumber(getFrameNumber() + 1);
}

void LayerIO::read(Buffer<float> &buffer, double &timestamp, int frameNumber) {
   setFrameNumber(frameNumber);
   read(buffer, timestamp);
}


void LayerIO::write(Buffer<float> const &buffer, double timestamp) {
   if (!mFileStream) { return; }

   long numBytes = checkBufferDimensions(buffer);
   mFileStream->write(&timestamp, 8L);
   mFileStream->write(buffer.asVector().data(), numBytes);
   setHeaderNBands();

   setFrameNumber(getFrameNumber() + 1);
   if (getFrameNumber() > getNumFrames()) {
      mNumFrames = getFrameNumber();
   }
}

void LayerIO::write(Buffer<float> const &buffer, double timestamp, int frameNumber) {
   setFrameNumber(frameNumber);
   write(buffer, timestamp);
}

void LayerIO::open() {
   mFileStream->open();
}

void LayerIO::close() {
   mFileStream->close();
}

void LayerIO::setFrameNumber(int frame) {
   if (!mFileStream) { return; }
   mFrameNumber = frame;
   long filePos = calcFilePositionFromFrameNumber(frame);
   mFileStream->setInPos(filePos, std::ios_base::beg);
   if (mFileStream->writeable()) { mFileStream->setOutPos(filePos, std::ios_base::beg); }
}

long LayerIO::calcValuesPerPVPFrame() const {
   long nf = static_cast<long>(mNumFeatures);
   long nx = static_cast<long>(mWidth);
   long ny = static_cast<long>(mHeight);
   long const valuesPerPVPFrame = nx * ny * nf;
   return valuesPerPVPFrame;
}

long LayerIO::checkBufferDimensions(Buffer<float> const &buffer) {
   int status = PV_SUCCESS;
   if (buffer.getWidth() != getWidth()) {
      ErrorLog().printf(
            "LayerIO received a buffer with width %d but file has width %d.\n",
            buffer.getWidth(), getWidth());
      status = PV_FAILURE;
   }
   if (buffer.getHeight() != getHeight()) {
      ErrorLog().printf(
            "LayerIO received a buffer with height %d but file has height %d.\n",
            buffer.getHeight(), getHeight());
      status = PV_FAILURE;
   }
   if (buffer.getFeatures() != getNumFeatures()) {
      ErrorLog().printf(
            "LayerIO received a buffer with %d features %d but file has %d features.\n",
            buffer.getFeatures(), getNumFeatures());
      status = PV_FAILURE;
   }
   FatalIf(status != PV_SUCCESS, "LayerIO \"%s\" failed.\n", getFileStream()->getFileName().c_str());

   long numBytes = static_cast<long>(buffer.getTotalElements()) * mDataSize;
   return numBytes;
}

void LayerIO::initializeNumFrames() {
   if (!mFileStream) { return; }

   getFileStream()->setInPos(0L, std::ios_base::end);
   long eofPosition = getFileStream()->getInPos();
   if (getFileStream()->writeable()) {
      if (eofPosition == 0L) { writeHeader(); }
      getFileStream()->setInPos(0L, std::ios_base::end);
      eofPosition = getFileStream()->getInPos();
      getFileStream()->setOutPos(eofPosition, std::ios_base::beg);
   }

   mNumFrames = calcFrameNumberFromFilePosition(eofPosition);
}

void LayerIO::setHeaderNBands() {
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

void LayerIO::writeHeader() {
   if (!getFileStream()) { return; }
   FatalIf(
         !getFileStream()->writeable(),
         "writeHeader() called but \"%s\" is not writeable.\n",
         getFileStream()->getFileName().c_str());

   BufferUtils::ActivityHeader header;
   header.headerSize = static_cast<int>(mHeaderSize);
   header.numParams  = 20;
   header.fileType   = PVP_NONSPIKING_ACT_FILE_TYPE;
   header.nx         = mWidth;
   header.ny         = mHeight;
   header.nf         = mNumFeatures;
   header.numRecords = 1;
   header.recordSize = header.nx * header.ny * header.nf;
   header.dataSize   = static_cast<int>(mDataSize);
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
