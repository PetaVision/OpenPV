/*
 * BroadcastPreWeightsIO.hpp
 *
 *  Created on: July 7, 2021
 *      Author: peteschultz
 */

#ifndef BROADCASTPREWEIGHTSIO_HPP_
#define BROADCASTPREWEIGHTSIO_HPP_

#include "io/FileStream.hpp"
#include "structures/Patch.hpp"
#include "structures/WeightData.hpp"
#include "utils/BufferUtilsPvp.hpp" // struct WeightHeader

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace PV {

/**
 * A class to manage weight PVP files with a broadcast layer as presynaptic input.
 * Generally reading and writing a weight PVP file with a broadcast pre-layer should be done by
 * means of this class. The principal use case is by the BroadcastPreWeightsFile class, which
 * creates and manages a BroadcastPreWeightsIO class internally for its I/O operations.
 *
 * The constructor takes a pointer to a FileStream, and information about the size of the weights.
 * Since the pre-layer is a broadcast layer, the NumDataPatchesX and NumDataPatchesY values must
 * both be 1. In addition, the patch size must be the entire post-synaptic layer.
 *
 * If the FileStream pointer points to an existing non-empty file, that file must be a local-patch
 * weights file with dimensions consistent with the constructor arguments.
 *
 * If the FileStream is read-only, the file position is moved to the beginning of the file.
 * Otherwise, it is moved to the end of the file.
 *
 * The number of frames in the file is retrieved by getNumFrames(). The current frame can be
 * set or retrieved using setFrameNumber() or getFrameNumber().
 * The file position can only be set to the beginning of a frame, Setting the position to index n
 * moves the file to the start of the (zero-indexed) nth frame.
 * For FileStreams in read/write mode, the read position and write position always move together.
 * If the file is read-only, setting FrameNumber to a value greater than NumFrames causes the
 * FrameNumber to be set to FrameNumber mod NumFrames.
 *
 * This class performs no MPI operations. However, a major use case is that this class run in
 * parallel in an MPIBlock, with the root process gathering from and scattering to the rest of the
 * processes. For this reason, the FileStream argument to the constructor may be the null pointer,
 * in which case file operations have no effect, set-methods are meaningless, and get-methods
 * return meaningless values. Additionally, the read and write operations are not atomic.
 *
 * To write a frame, one should call the following functions:
 * * setHeaderTimestamp() to set the timestamp field of the frame's header
 * * setHeaderMinVal() to set the minVal field of the frame's header
 * * setHeaderMaxVal() to set the maxVal field of the frame's header
 *      (setHeaderExtremeVal() can be used to set minVal and maxVal together)
 * * writeHeader() to write the header to the current frame
 * * writeRegion() one or more times, with each call writing part or all of the frame's data
 * * finishWrite() to increment the FrameNumber and set the file position to the start of the
 *      new frame (which may be end of file)
 *
 * To read a frame, one should call the following functions:
 * * readHeader() to read the header. The header is not returned, but the values of the header
 *      are sanity-checked and the timestamp, minVal, and maxVal fields are made available.
 * * getHeaderTimestamp(), getHeaderMinVal(), getHeaderMaxVal() if these header fields are needed.
 * * readRegion() one or more times, with each call reading part or all of the frame's data
 * * finishRead() to incement the FrameNumber and set the file position to the start of the
 *      new frame (which may be the end of file).
 */
class BroadcastPreWeightsIO {
  public:
   /**
    * BroadcastPreWeightsIO
    * fileStream is a pointer to a FileStream object. If nullptr, file operations have no effect.
    *    If non-null, the pointer should point to an existing file; if non-empty the file
    *    should be a local-patch weights file with header fields consistent with the other
    *    constructor argurments.
    * patchSizeX is the nxp argument, which corresponds to the width of the post-layer.
    * patchSizeY is the nyp argument, which corresponds to the height of the post-layer.
    * patchSizeF is the nfp argument, which corresponds to the number of post features.
    * nfPre      is the nf argument, which corresponds to the number of pre features
    *    (nxPre and nyPre are both 1, since the pre-layer must be a broadcast layer for this class).
    * numArbors is the number of arbors in the connection.
    * compressedFlag indicates whether the weights are stored in uint8-compressed format or not.
    */
   BroadcastPreWeightsIO(
         std::shared_ptr<FileStream> fileStream,
         int patchSizeX,
         int patchSizeY,
         int patchSizeF,
         int nfPre,
         int numArbors,
         bool compressedFlag);

   virtual ~BroadcastPreWeightsIO() {}

   /**
    * Calculate the file position from the indicated frameNumber, which is the frameNumber times
    * the size of one frame in bytes (see the private function member calcFrameSizeBytes()).
    * @details This is useful when writing checkpoints, because we checkpoint the file position
    * as a byte offset, not the frame number.
    */
   long calcFilePositionFromFrameNumber(int frameNumber) const;

   /**
    * Calculate the frame number from the indicated filePosition, which is the filePosition
    * divided by the size of one frame in bytes, discarding any remainder.
    * For the frame size, see the private function member calcFrameSizeBytes().
    * @details This is useful when reading checkpoints, because we checkpoint the file position
    * as a byte offset, not the frame number.
    */
   int calcFrameNumberFromFilePosition(long filePosition) const;

   /**
    * Finishes writing a frame. Increments frame number, and if the new frame is at the end of
    * the file, increments the number of frames.
    */
   void finishWrite();

   /**
    * Reads the header of the current frame into memory. It checks that the header fields are
    * correct for the patch size and number of presynaptic features specified in the constructor,
    * and errors out if they are not correct. The only fields not determened by the constructor
    * arguments are timestamp, minVal, and maxVal. These values are now available through the
    * corresponding getHeader-methods.
    */
   void readHeader();

   /**
    * Sets the frame number to the indicated value, and then calls readHeader() with no arguments.
    */
   void readHeader(int frameNumber);

   /**
    * Reads a portion of the data from the current frame into the indicated WeightData object.
    * the interval [xStart, xStart + weightData.getPatchSizeX()] must be a subset of
    *    the interval [0, PatchSizeX].
    * [yStart, yStart + weightData.getPatchSizeY()] must be contained in [0, PatchSizeY].
    * [fStart, fStart + weightData.getPatchSizeF()] must be contained in [0, PatchSizeF].
    * [fPreStart, fPreStart + weightData.getNumDataPatchesF()] must be contained in [0, NfPre]. 
    * [arborIndexStart, arborIndexStart + weightData.getNumArbors()] must be contained in
    *    the interval [0, NumArbors].
    */
   void readRegion(
         WeightData &weightData,
         int xStart,
         int yStart,
         int fStart,
         int fPreStart,
         int arborIndexStart);

   /**
    * Uses the current values of timestamp, minVal, and maxVal (whether they were set by calling
    * the setHeader-methods or by an earlier read) to write the header to the current frame.
    */
   void writeHeader();

   /**
    * Sets the frame number to the indicated value, and then calls writeHeader() without arguments.
    */
   void writeHeader(int frameNumber);

   /**
    * Writes the data in the indicated WeightData object to a portion of the current frame's data.
    * the interval [xStart, xStart + weightData.getPatchSizeX()] must be a subset of
    *    the interval [0, PatchSizeX].
    * [yStart, yStart + weightData.getPatchSizeY()] must be contained in [0, PatchSizeY].
    * [fStart, fStart + weightData.getPatchSizeF()] must be contained in [0, PatchSizeF].
    * [fPreStart, fPreStart + weightData.getNumDataPatchesF()] must be contained in [0, NfPre]. 
    * [arborIndexStart, arborIndexStart + weightData.getNumArbors()] must be contained in
    *    the interval [0, NumArbors].
    */
   void writeRegion(
         WeightData const &weightData,
         int xStart,
         int yStart,
         int fStart,
         int fPreStart,
         int arborIndexStart);

   /** Returns a pointer to the FileStream object managed by the BroadcastPreWeightsIO object.  */
   std::shared_ptr<FileStream> getFileStream() const { return mFileStream; }

   void open();
   void close();

   /** Returns the PatchSizeX value set by the constructor */
   int getPatchSizeX() const { return mPatchSizeX; }

   /** Returns the PatchSizeY value set by the constructor */
   int getPatchSizeY() const { return mPatchSizeY; }

   /** Returns the PatchSizeY value set by the constructor */
   int getPatchSizeF() const { return mPatchSizeF; }

   /**
    * Returns the overall patch size (PatchSizeX * PatchSizeY * PatchSizeF), determined by the
    * values set by the constructor.
    */
   long getPatchSizeOverall() const {
      return static_cast<long>(mPatchSizeX * mPatchSizeY * mPatchSizeF);
   }


   /**
    * Returns the NfPre value set by the constructor, that is, the number of presynaptic features.
    */
   int getNfPre() const { return mNfPre; }

   /** Returns the number of arbors, as set by the constructor */
   int getNumArbors() const { return mNumArbors; }

   /** Returns the compressed flag, as set by the constructor */
   bool getCompressedFlag() const { return mCompressedFlag; }

   long int getFrameSize() const { return mFrameSize; }

   /** Returns the current frame number */
   int getFrameNumber() const { return mFrameNumber; }

   /** Sets the frame number to the indicated value */
   void setFrameNumber(int frame);

   /** Returns the number of frames in the file */
   int getNumFrames() const { return mNumFrames; }

   /** Returns the current value of minVal in the header. */
   float getHeaderMinVal() const { return mHeader.minVal; }

   /** Returns the current value of maxVal in the header. */
   float getHeaderMaxVal() const { return mHeader.maxVal; }

   /** Returns the current timestamp in the header. */
   double getHeaderTimestamp() const { return mHeader.baseHeader.timestamp; }

   /** Resets the header's minVal to -M and maxVal to +M, where M has large absolute value */
   void resetHeaderExtremeVals() {
      mHeader.minVal = std::numeric_limits<float>::lowest();
      mHeader.maxVal = std::numeric_limits<float>::max();
   }

   /** Sets the header's minVal and maxVal to the indicated values */
   void setHeaderExtremeVals(double minVal, double maxVal) {
      mHeader.minVal = minVal;
      mHeader.maxVal = maxVal;
   }

   /** Sets the header's minVal to the indicated value */
   void setHeaderMinVal(double minVal) { mHeader.minVal = minVal; }

   /** Sets the header's maxVal to the indicated value */
   void setHeaderMaxVal(double maxVal) { mHeader.maxVal = maxVal; }

   /** Sets the header's timestamp to the indicated value */
   void setHeaderTimestamp(double timestamp) { mHeader.baseHeader.timestamp = timestamp; }

  private:
   long calcArborSizeBytes() const;
   long calcFrameSizeBytes() const;
   long calcPatchSizeBytes() const;
   static std::array<std::vector<int>, 2> calcPatchStartsAndStops(
         int nExtendedPre,
         int nRestrictedPre,
         int nPreRef,
         int nPostRef,
         int patchSize);
   int checkHeader(BufferUtils::WeightHeader const &header) const;
   int checkHeaderField(
         int expected, int observed, std::string const &fieldLabel, int oldStatus) const;
   int checkHeaderField(
         double expected, double observed, std::string const &fieldLabel, int oldStatus) const;

   void initializeFrameSize();
   void initializeHeader();
   void initializeNumFrames();

  private:
   std::shared_ptr<FileStream> mFileStream;
   int mPatchSizeX;
   int mPatchSizeY;
   int mPatchSizeF;
   int mNfPre;
   int mNumArbors;
   bool mCompressedFlag;

   long mFrameSize  = 0L; // Number of bytes in a frame, including the header
   int mFrameNumber = 0;
   int mNumFrames   = 0;

   long mDataSize              = static_cast<long>(sizeof(float));
   long const mHeaderSize      = static_cast<long>(sizeof(BufferUtils::WeightHeader));
   long const mPatchHeaderSize = static_cast<long>(sizeof(Patch));

   BufferUtils::WeightHeader mHeader;
   bool mHeaderWrittenFlag = false;
};

} // namespace PV

#endif // BROADCASTPREWEIGHTSIO_HPP_
