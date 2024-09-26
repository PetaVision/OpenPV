/*
 * BroadcastPreWeightsFile.hpp
 */

#ifndef BROADCASTPREWEIGHTSFILE_HPP_
#define BROADCASTPREWEIGHTSFILE_HPP_

#include "checkpointing/CheckpointerDataInterface.hpp"
#include "io/FileManager.hpp"
#include "io/BroadcastPreWeightsIO.hpp"
#include "io/WeightsFile.hpp"
#include "utils/BufferUtilsPvp.hpp" // WeightHeader

namespace PV {

/**
 * A class to manage PVP files for weights where the pre-synaptic input is a broadcast layer.
 * It handles all MPI gather/scatter operations, M-to-N communication, and PVP file format details.
 * All file operations treat the connection state, i.e. all weights at a single timestep, as a unit.
 *
 * The file position can only be set to the beginning of a frame, Setting the position to index n
 * moves the file to the start of the (zero-indexed) nth frame.
 * For files opened in read/write mode, the read position and write position always move together.
 */
class BroadcastPreWeightsFile : public WeightsFile {
  public:
   BroadcastPreWeightsFile(
         std::shared_ptr<FileManager const> fileManager,
         std::string const &path,
         std::shared_ptr<WeightData> weightData,
         int nfPre,
         bool compressedFlag,
         bool readOnlyFlag,
         bool clobberFlag,
         bool verifyWrites);

   BroadcastPreWeightsFile() = delete;

   ~BroadcastPreWeightsFile();

   virtual void read() override;
   virtual void read(double &timestamp) override;
   virtual void write(double timestamp) override;

   void truncate(int index) override;

   std::string const &getPath() const { return mPath; }

   int getPatchSizePerProcX() const { return mPatchSizePerProcX; }
   int getPatchSizePerProcY() const { return mPatchSizePerProcY; }
   int getPatchSizeF() const { return mPatchSizeF; }
   long getPatchSizeOverall() const {
      return mPatchSizePerProcX * mPatchSizePerProcY * mPatchSizeF;
   }
   int getNfPre() const { return mNfPre; }
   // nfRestrictedPost would be the same as patchSizeF
   int getNumArbors() const { return mNumArbors; }
   bool getCompressedFlag() const { return mCompressedFlag; }
   bool getReadOnly() const { return mReadOnly; }
   bool getVerifyWrites() const { return mVerifyWrites; }
   int getNumFrames() const { return mBroadcastPreWeightsIO->getNumFrames(); }

   void setIndex(int index) override;

   std::shared_ptr<FileStream> getFileStream() const {
      return mBroadcastPreWeightsIO->getFileStream();
   }

  protected:
   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   virtual Response::Status processCheckpointRead(double simTime) override;

  private:
   int initializeCheckpointerDataInterface();
   void initializeBroadcastPreWeightsIO(bool clobberFlag);

   bool isRoot() { return mFileManager->isRoot(); }

   void readInternal(double &timestamp);

   BufferUtils::WeightHeader createHeader(double timestamp, float minWgt, float maxWgt) const;

  private:
   std::shared_ptr<FileManager const> mFileManager;
   std::string mPath;
   int mPatchSizePerProcX;
   int mPatchSizePerProcY;
   int mPatchSizeF;
   int mNfPre;
   // mNfRestrictedPost would be the same as patchSizeF
   // mNumPatchesF is the same as mNfPre; mNumPatchesX and mNumPatchesY are always 1
   int mNumArbors;
   bool mCompressedFlag;
   bool mReadOnly;
   bool mVerifyWrites;

   std::unique_ptr<BroadcastPreWeightsIO> mBroadcastPreWeightsIO;

   long mFileStreamReadPos  = 0L; // Input file position of the BroadcastPreWeightsIO's FileStream
   long mFileStreamWritePos = 0L; // Output file position of the BroadcastPreWeightsIO's FileStream
};

} // namespace PV

#endif // BROADCASTPREWEIGHTSFILE_HPP_
