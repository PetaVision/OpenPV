/*
 * LocalPatchWeightsIO.hpp
 *
 *  Created on: July 7, 2021
 *      Author: peteschultz
 */

#ifndef WEIGHTSIO_HPP_
#define WEIGHTSIO_HPP_

#include "io/FileStream.hpp"
#include "utils/BufferUtilsPvp.hpp" // struct WeightHeader
#include "structures/Buffer.hpp"
#include "structures/Patch.hpp"
#include "structures/WeightData.hpp"

#include <array>
#include <memory>
#include <vector>

namespace PV {

/**
 * A class to manage weight PVP files. 
 * Generally reading and writing a weight PVP file should be done by means of this class.
 * The principal use case is by the LocalPatchWeightsFile class, which creates and manages a
 * LocalPatchWeightsIO class internally for its I/O operations.
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
         int nxRestrictedRegion, int nyRestrictedRegion,
         float &minWeight, float &maxWeight) const;
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
      int regionNxRestricted,
      int regionNyRestricted,
      int regionXStartRestricted,
      int regionYStartRestricted,
      int regionFStartRestricted,
      int arborIndexStart);
  
   void writeHeader(BufferUtils::WeightHeader const &header);
   void writeHeader(BufferUtils::WeightHeader const &header, int frameNumber);
   void writeRegion(
      WeightData const &weightData,
      BufferUtils::WeightHeader const &header,
      int regionNxRestricted,
      int regionNyRestricted,
      int regionXStartRestricted,
      int regionYStartRestricted,
      int regionFStartRestricted,
      int arborIndexStart);

   std::shared_ptr<FileStream> getFileStream() const { return mFileStream; }

   int getPatchSizeX() const { return mPatchSizeX; }
   int getPatchSizeY() const { return mPatchSizeY; }
   int getPatchSizeF() const { return mPatchSizeF; }
   long getPatchSizeOverall() const {
      return static_cast<long>(mPatchSizeX * mPatchSizeY * mPatchSizeF);
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

   long int getFrameSize() const { return mFrameSize; }
   int getFrameNumber() const { return mFrameNumber; }
   void setFrameNumber(int frame);
   int getNumFrames() const { return mNumFrames; }

   int getXMargin() const { return mXMargin; }
   int getYMargin() const { return mYMargin; }

  private:
   long calcArborSizeBytes() const;
   long calcFrameSizeBytes() const;
   long calcPatchSizeBytes() const;
   static std::array<std::vector<int>, 2> calcPatchStartsAndStops(
         int nExtendedPre, int nRestrictedPre, int nPreRef, int nPostRef, int patchSize);
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
   void initializeNumFrames();

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

   void writePatchAtLocation(
      std::vector<float> const &writeBuffer,
      int xStart, int xStop, int yStart, int yStop, float minWeight, float maxWeight);

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

   long mFrameSize  = 0L; // Number of bytes in a frame, including the header
   int mFrameNumber     = 0;
   int mNumFrames       = 0;

   int mXMargin = 0;
   int mYMargin = 0;

   long mDataSize = static_cast<long>(sizeof(float));
   long const mHeaderSize = static_cast<long>(sizeof(BufferUtils::WeightHeader));
   long const mPatchHeaderSize = static_cast<long>(sizeof(Patch));
};

} // namespacePV

#endif // WEIGHTSIO_HPP_
