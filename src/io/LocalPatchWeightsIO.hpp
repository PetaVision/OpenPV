/*
 * LocalPatchWeightsIO.hpp
 *
 *  Created on: July 7, 2021
 *      Author: peteschultz
 */

#ifndef LOCALPATCHWEIGHTSIO_HPP_
#define LOCALPATCHWEIGHTSIO_HPP_

#include "io/FileStream.hpp"
#include "io/PVPFrameIndexer.hpp"
#include "structures/Patch.hpp"
#include "structures/WeightData.hpp"
#include "utils/BufferUtilsPvp.hpp" // struct WeightHeader

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace PV {

/**
 * A class to manage weight PVP files.
 * Generally reading and writing a local-patch weight PVP file should be done by means of this
 * class. The principal use case is by the LocalPatchWeightsFile class, which creates and manages
 * a LocalPatchWeightsIO class internally for its I/O operations.
 *
 * Opening a new file using LocalPatchWeightsIO creates an empty file. The public function members
 * read() loads all the weights from the frame. The writeHeader() function writes the header.
 * The writeRegion() function reads a region of the frame.
 *
 * The file position can only be set to the beginning of a frame, Setting the position to index n
 * moves the file to the start of the (zero-indexed) nth frame.
 * For files opened in read/write mode, the read position and write position always move together.
 *
 * The buffers that make up the arbors can have greater dimensions than is required by the patch
 * sizes and fileExtended flags. In this case, only the required part of the buffer is written/read.
 * It is assumed that the left and right margins have the same size; similarly the up and down
 * margins. The horizontal and vertical margins can have different sizes from each other.
 * It is an error for the buffer width and NxRestrictedPre to differ by an odd amount; similarly
 * for the buffer height and NyRestrictedPre. Also, the margins must be at least as large as
 * required by the patch sizes and FileExtended setting.
 */
class LocalPatchWeightsIO {
  public:
   LocalPatchWeightsIO(
         std::shared_ptr<FileStream> fileStream,
         int patchSizeX,
         int patchSizeY,
         int patchSizeF,
         int nxRestrictedPre,
         int nyRestrictedPre,
         int nfRestrictedPre,
         int nxRestrictedPost,
         int nyRestrictedPost,
         // nfRestrictedPost would be the same as patchSizeF
         int numArbors,
         bool fileExtendedFlag,
         bool compressedFlag);

   virtual ~LocalPatchWeightsIO() {}

   /**
    * Calculates the minimum and maximum weights in the active parts of
    * a WeightData object. The WeightData is assumed to be a region of
    * weights from a connection with the same pre/post ratios as the current
    * object. If the FileExtended flag is true, the margins of the WeightData
    * object must be at least as large as required by the current object's
    * pre/post ratios and patch sizes. The size of the restricted layer of the
    * WeightData object's presynaptic layer must be given, because it is not
    * necessarily the same as the current object's NxRestrictedPre and
    * NyRestrictedPre.
    */
   void calcExtremeWeights(
         WeightData const &weightRegion,
         int nxPreRestrictedRegion,
         int nyPreRestrictedRegion,
         int nxPostRestrictedRegion,
         int nyPostRestrictedRegion,
         float &minWeight,
         float &maxWeight) const;
   long calcFilePositionFromFrameNumber(int frameNumber) const;
   int calcFrameNumberFromFilePosition(long filePosition) const;

   /**
    * Increments frame number, and if necessary, increments the number of frames.
    * If necessary (because of shrunken patches), pads the file with zeroes to
    * complete the last frame.
    */
   void finishWrite();

   BufferUtils::WeightHeader readHeader();
   BufferUtils::WeightHeader readHeader(int frameNumber);
   void readRegion(
         WeightData &weightData,
         BufferUtils::WeightHeader const &header,
         int regionNxRestrictedPre,
         int regionNyRestrictedPre,
         int regionNxRestrictedPost,
         int regionNyRestrictedPost,
         int regionXStartRestricted,
         int regionYStartRestricted,
         int regionFStartRestricted,
         int arborIndexStart);

   void writeHeader(BufferUtils::WeightHeader const &header);
   void writeHeader(BufferUtils::WeightHeader const &header, int frameNumber);
   void writeRegion(
         WeightData const &weightData,
         BufferUtils::WeightHeader const &header,
         int regionNxRestrictedPre,
         int regionNyRestrictedPre,
         int regionNxRestrictedPost,
         int regionNyRestrictedPost,
         int regionXStartRestricted,
         int regionYStartRestricted,
         int regionFStartRestricted,
         int arborIndexStart);

   std::shared_ptr<FileStream> getFileStream() const { return mFileStream; }

   void open();
   void close();

   int getPatchSizeX() const { return mPatchSizeX; }
   int getPatchSizeY() const { return mPatchSizeY; }
   int getPatchSizeF() const { return mPatchSizeF; }
   long getPatchSizeOverall() const {
      return (long)mPatchSizeX * (long)mPatchSizeY * (long)mPatchSizeF;
   }

   int getNxRestrictedPre() const { return mNxRestrictedPre; }
   int getNyRestrictedPre() const { return mNyRestrictedPre; }
   int getNfPre() const { return mNfPre; }
   int getNxRestrictedPost() const { return mNxRestrictedPost; }
   int getNyRestrictedPost() const { return mNyRestrictedPost; }
   int getNumArbors() const { return mNumArbors; }
   bool getFileExtendedFlag() const { return mFileExtendedFlag; }
   bool getCompressedFlag() const { return mCompressedFlag; }

   long getNumPatchesFile() const;

   /** Returns the current frame number */
   int getFrameNumber() const { return mFrameIndexer->getFrameNumber(); }

   /** Sets the frame number to the indicated value */
   void setFrameNumber(int frame) { mFrameIndexer->setFrameNumber(frame); }

   /** Returns the number of frames in the file */
   int getNumFrames() const { return mFrameIndexer->getNumFrames(); }

   int getXMargin() const { return mXMargin; }
   int getYMargin() const { return mYMargin; }

  private:
   long calcArborSizeBytes() const;
   long calcFrameSizeBytes() const;
   long calcPatchSizeBytes() const;

   /**
    * @brief calculates the 1-D locations of starting and stopping indices of
    * shrunken patches, in postsynaptic patch space.
    * @details Inputs:
    *    nExtendedPre     local dimensions of extended presynaptic space
    *    nRestrictedPre   local dimensions of restricted presynaptic space
    *    nRestrictedPost  local dimensions of restricted postsynaptic space
    *    nPreRef          global dimensions of restricted presynaptic space
    *    nPostRef         global dimensions of restricted postsynaptic space
    *    patchSize        patch size in postsynaptic space
    */
   static std::array<std::vector<int>, 2> calcPatchStartsAndStops(
         int nExtendedPre,
         int nRestrictedPre,
         int nRestrictedPost,
         int nPreRef,
         int nPostRef,
         int patchSize);

   /**
    * checkDimensions(
    *       weightData,
    *       regionNxRestrictedPre,
    *       regionNyRestrictedPre,
    *       regionXStartRestricted,
    *       regionYStartRestricted,
    *       regionFStartRestricted,
    *       arborIndexStart,
    *       functionName)
    * Checks whether a WeightData object fits into the LocalPatchWeightsIO object's weight data.
    * Specifically, it confirms that
    *     weightData.getNumArbors() + arborIndexStart <= getNumArbors().
    *
    * If weightData object does not fit, an error message is printed and the run exits with a
    * fatal error.
    */
   void checkDimensions(
         WeightData const &weightData,
         int regionNxRestrictedPre,
         int regionNyRestrictedPre,
         int regionXStartRestricted,
         int regionYStartRestricted,
         int regionFStartRestricted,
         int arborIndexStart,
         std::string const &functionName);
   void checkHeader(BufferUtils::WeightHeader const &header) const;

   void initializeMargins();
   void initializeFrameIndexer();

   void readPatch(
         std::vector<float> &readBuffer,
         int arborIndex,
         int xPatchIndex,
         int yPatchIndex,
         int fPatchIndex,
         float minWeight,
         float maxWeight);

   void writePatch(
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
         float maxWeight);
   // void setHeaderNBands(); // We might do this for weights as we do for layers; for now we don't

   /**
    * writePatchAtLocation(buffer, xStart, xStop, yStart, yStop, minWeight, maxWeight)
    * Writes a buffer of length (xStop-xStart)*(yStop-yStart)*PatchSizeF to the FileStream,
    * to the region indicated by the indices xStart, xStop, yStart, yStop. All features are written.
    *
    * Before calling this function, the FileStream position must be set to the start of the patch
    * data (the end of the patch's 8-byte header) of the patch where the data is to be written.
    * On exit, the FileStream position is the same as it was on entry.
    *
    * If CompressedFlag is true, the input buffer still consists of float values, and the
    * compression is performed inside this function.
    */
   void writePatchAtLocation(
         std::vector<float> const &writeBuffer,
         int xStart,
         int xStop,
         int yStart,
         int yStop,
         float minWeight,
         float maxWeight);

  private:
   std::shared_ptr<FileStream> mFileStream;
   int mPatchSizeX;
   int mPatchSizeY;
   int mPatchSizeF;
   int mNxRestrictedPre;
   int mNyRestrictedPre;
   int mNfPre;
   int mNxRestrictedPost;
   int mNyRestrictedPost;
   int mNumArbors;
   bool mFileExtendedFlag;
   bool mCompressedFlag;

   std::shared_ptr<PVPFrameIndexer> mFrameIndexer = nullptr;

   int mXMargin = 0;
   int mYMargin = 0;

   long mDataSize              = static_cast<long>(sizeof(float));
   long const mHeaderSize      = static_cast<long>(sizeof(BufferUtils::WeightHeader));
   long const mPatchHeaderSize = static_cast<long>(sizeof(Patch));
};

} // namespace PV

#endif // LOCALPATCHWEIGHTSIO_HPP_
