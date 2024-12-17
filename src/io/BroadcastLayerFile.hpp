#ifndef BROADCASTLAYERFILE_HPP_
#define BROADCASTLAYERFILE_HPP_

#include "checkpointing/CheckpointerDataInterface.hpp"
#include "io/FileManager.hpp"
#include "io/LayerIO.hpp"

#include <memory>
#include <string>

namespace PV {

/**
 * A class to manage dense activity PVP files for broadcast layers. It internally handles all
 * MPI gather/scatter operations, M-to-N communication, and PVP file format details. All file
 * operations treat the layer state, i.e. the data of all batch elements at a single timestep,
 * as a unit.
 */
class BroadcastLayerFile : public CheckpointerDataInterface {
  public:
   BroadcastLayerFile(
         std::shared_ptr<FileManager const> fileManager,
         std::string const &path,
         int numFeatures,
         int localBatchWidth,
         bool readOnlyFlag,
         bool clobberFlag,
         bool verifyWrites);

   BroadcastLayerFile() = delete;

   virtual ~BroadcastLayerFile();

   void read();
   void read(double &timestamp);
   void write(double timestamp);

   void truncate(int index);

   int getIndex() const { return mIndex; }
   void setIndex(int index);

   std::string const &getPath() const { return mPath; }

   float const *getDataLocation(int index) const { return mDataLocations.at(index); }
   float *getDataLocation(int index) { return mDataLocations.at(index); }
   void setDataLocation(float *location, int index) { mDataLocations.at(index) = location; }

  private:
   void initializeLayerIO(bool clobberFlag);
   void readInternal(double &timestamp, bool checkTimestampConsistency);

  private:
   std::shared_ptr<FileManager const> mFileManager = nullptr;
   std::string mPath;
   int mNumFeatures;
   int mLocalBatchWidth;
   bool mReadOnly;
   bool mVerifyWrites;

   int mIndex = 0;
   std::vector<float *> mDataLocations;

   std::unique_ptr<LayerIO> mLayerIO;

   int mNumFrames = 0; // number of framesets in the file.
   // (A frameset is a set of PVP frames written or read together; the number of frames in the
   // frameset is MPIBlock->BatchDimension * mLocalBatchWidth

   // The following values are written during checkpointing.
   // It would be more logical to write the value of mIndex, but for reasons of
   // backward compatibility, we continue to write the old values.
   // that is, Index * MPIBlock->BatchDimension * mLocalBatchWidth
   long mFileStreamReadPos  = 0L; // Input file position of the LayerIO's FileStream
   long mFileStreamWritePos = 0L; // Output file position of the LayerIO's FileStream
};

} // namespace PV

#endif // BROADCASTLAYERFILE_HPP_
