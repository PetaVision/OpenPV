/*
 * LocalPatchWeightsFile.hpp
 *
 *  Created on: Jun 30, 2021
 *      Author: peteschultz
 */

#ifndef LOCALPATCHWEIGHTSFILE_HPP_
#define LOCALPATCHWEIGHTSFILE_HPP_

#include "checkpointing/CheckpointerDataInterface.hpp"
#include "io/FileManager.hpp"
#include "io/LocalPatchWeightsIO.hpp"
#include "structures/WeightData.hpp"
#include "utils/BufferUtilsPvp.hpp" // WeightHeader

namespace PV {

/**
 * A class to manage local patch weights (i.e. non-shared) PVP files. It internally handles all
 * MPI gather/scatter operations, M-to-N communication, and PVP file format details. All file
 * operations treat the connection state, i.e. all weights at a single timestep, as a unit.
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
class LocalPatchWeightsFile : public CheckpointerDataInterface{
  public:
   LocalPatchWeightsFile(
      std::shared_ptr<FileManager const> fileManager,
      std::string const &path,
      std::shared_ptr<WeightData> weightData,
      PVLayerLoc const *preLayerLoc,
      PVLayerLoc const *postLayerLoc,
      bool fileExtendedFlag,
      bool compressedFlag,
      bool readOnlyFlag,
      bool clobberFlag,
      bool verifyWrites);

   LocalPatchWeightsFile() = delete;

   ~LocalPatchWeightsFile();

   void read(WeightData &weightData);
   void read(WeightData &weightData, double &timestamp);
   void write(WeightData &weightData, double timestamp);

   void truncate(int index);

   std::string const &getPath() const { return mPath; }

   int getPatchSizeX() const { return mPatchSizeX; }
   int getPatchSizeY() const { return mPatchSizeY; }
   int getPatchSizeF() const { return mPatchSizeF; }
   long getPatchSizeOverall() const { return mPatchSizeX * mPatchSizeY * mPatchSizeF; }
   int getNumPatchesX() const { return mNumPatchesX; }
   int getNumPatchesY() const { return mNumPatchesY; }
   int getNumPatchesF() const { return mNumPatchesF; }
   long getNumPatchesOverall() const { return mNumPatchesX * mNumPatchesY * mNumPatchesF; }
   int getNxRestrictedPre() const { return mNxRestrictedPre; }
   int getNyRestrictedPre() const { return mNyRestrictedPre; }
   int getNfPre() const { return mNfPre; }
   int getNxRestrictedPost() const { return mNxRestrictedPost; }
   int getNyRestrictedPost() const { return mNyRestrictedPost; }
   // nfRestrictedPost would be the same as patchSizeF
   int getNumArbors() const { return mNumArbors; }
   bool getFileExtendedFlag() const { return mFileExtendedFlag; }
   bool getCompressedFlag() const { return mCompressedFlag; }
   bool getReadOnly() const { return mReadOnly; }
   bool getVerifyWrites() const { return mVerifyWrites; }

   int getIndex() const { return mIndex; }
   void setIndex(int index);

   std::shared_ptr<FileStream> getFileStream() const {
         return mLocalPatchWeightsIO->getFileStream();
   }

  protected:
   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   virtual Response::Status processCheckpointRead() override;

  private:
   int initializeCheckpointerDataInterface();
   void initializeLocalPatchWeightsIO(bool clobberFlag);

   bool isRoot() { return mFileManager->isRoot(); }

   void readInternal(WeightData &weightData, double &timestamp);

   BufferUtils::WeightHeader createHeader(double timestamp, float minWgt, float maxWgt) const;

  private:
   std::shared_ptr<FileManager const> mFileManager;
   std::string mPath;
   int mPatchSizeX;
   int mPatchSizeY;
   int mNxRestrictedPre;
   int mNyRestrictedPre;
   int mNfPre;
   int mNxRestrictedPost;
   int mNyRestrictedPost;
   // mNfRestrictedPost would be the same as patchSizeF
   int mPatchSizeF;
   int mNumPatchesX;
   int mNumPatchesY;
   int mNumPatchesF;
   int mNumArbors;
   bool mFileExtendedFlag;
   bool mCompressedFlag;
   bool mReadOnly;
   bool mVerifyWrites;

   int mIndex = 0;

   std::unique_ptr<LocalPatchWeightsIO> mLocalPatchWeightsIO;

   long mFileStreamReadPos  = 0L; // Input file position of the LayerIO's FileStream
   long mFileStreamWritePos = 0L; // Output file position of the LayerIO's FileStream
};

} // namespacePV

#endif // LOCALPATCHWEIGHTSFILE_HPP_
