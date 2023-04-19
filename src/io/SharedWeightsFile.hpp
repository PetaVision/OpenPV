/*
 * SharedWeightsFile.hpp
 *
 *  Created on: Jun 30, 2021
 *      Author: peteschultz
 */

#ifndef SHAREDWEIGHTSFILE_HPP_
#define SHAREDWEIGHTSFILE_HPP_

#include "checkpointing/CheckpointerDataInterface.hpp"
#include "io/FileManager.hpp"
#include "io/SharedWeightsIO.hpp"
#include "io/WeightsFile.hpp"

namespace PV {

/**
 * A class to manage shared-weights PVP files. It internally handles all MPI broadcast
 * operations, M-to-N communication, and PVP file format details. All file operations treat
 * the connection state, i.e. all weights at a single timestep, as a unit.
 */
class SharedWeightsFile : public WeightsFile {
  public:
   SharedWeightsFile(
         std::shared_ptr<FileManager const> fileManager,
         std::string const &path,
         std::shared_ptr<WeightData> weightData,
         bool compressedFlag,
         bool readOnlyFlag,
         bool clobberFlag,
         bool verifyWrites);

   SharedWeightsFile() = delete;

   ~SharedWeightsFile();

   void read(WeightData &weightData) override;
   void read(WeightData &weightData, double &timestamp) override;
   void write(WeightData const &weightData, double timestamp) override;

   void truncate(int index) override;

   std::string const &getPath() const { return mPath; }

   int getPatchSizeX() const { return mPatchSizeX; }
   int getPatchSizeY() const { return mPatchSizeY; }
   int getPatchSizeF() const { return mPatchSizeF; }
   long getPatchSizeOverall() const { return mPatchSizeX * mPatchSizeY * mPatchSizeF; }
   int getNumPatchesX() const { return mNumPatchesX; }
   int getNumPatchesY() const { return mNumPatchesY; }
   int getNumPatchesF() const { return mNumPatchesF; }
   long getNumPatchesOverall() const { return mNumPatchesX * mNumPatchesY * mNumPatchesF; }
   int getNumArbors() const { return mNumArbors; }
   bool getCompressedFlag() const { return mCompressedFlag; }
   bool getReadOnly() const { return mReadOnly; }
   bool getVerifyWrites() const { return mVerifyWrites; }

   void setIndex(int index) override;

   std::shared_ptr<FileStream> getFileStream() const { return mSharedWeightsIO->getFileStream(); }

  protected:
   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   virtual Response::Status processCheckpointRead(double simTime) override;

  private:
   int initializeCheckpointerDataInterface();
   void initializeSharedWeightsIO(bool clobberFlag);

   bool isRoot() { return mFileManager->isRoot(); }

   void readInternal(WeightData &weightData, double &timestamp);

  private:
   std::shared_ptr<FileManager const> mFileManager;
   std::string mPath;
   int mPatchSizeX;
   int mPatchSizeY;
   int mPatchSizeF;
   int mNumPatchesX;
   int mNumPatchesY;
   int mNumPatchesF;
   int mNumArbors;
   bool mCompressedFlag;
   bool mReadOnly;
   bool mVerifyWrites;

   std::unique_ptr<SharedWeightsIO> mSharedWeightsIO;

   long mFileStreamReadPos  = 0L; // Input file position of the SharedWeightsIO's FileStream
   long mFileStreamWritePos = 0L; // Output file position of the SharedWeightsIO's FileStream
};

} // namespace PV

#endif // SHAREDWEIGHTSFILE_HPP_
