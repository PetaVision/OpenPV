/*
 * LayerFile.hpp
 *
 *  Created on: May 3, 2021
 *      Author: peteschultz
 */

#ifndef LAYERFILE_HPP_
#define LAYERFILE_HPP_

#include "checkpointing/CheckpointerDataInterface.hpp"
#include "include/PVLayerLoc.hpp"
#include "io/FileManager.hpp"
#include "io/LayerBatchGatherScatter.hpp"
#include "io/LayerIO.hpp"

#include <memory>
#include <string>

namespace PV {

/**
 * A class to manage dense activity PVP files. It internally handles all MPI gather/scatter
 * operations, M-to-N communication, and PVP file format details. All file operations treat
 * the layer state, i.e. the data of all batch elements at a single timestep, as a unit.
 */
class LayerFile : public CheckpointerDataInterface {
  public:
   LayerFile(
         std::shared_ptr<FileManager const> fileManager,
         std::string const &path,
         PVLayerLoc const &layerLoc,
         bool dataExtendedFlag,
         bool fileExtendedFlag,
         bool readOnlyFlag,
         bool clobberFlag,
         bool verifyWrites);

   LayerFile() = delete;

   ~LayerFile();

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

  protected:
   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;

   virtual Response::Status processCheckpointRead(double simTime) override;

  private:
   int initializeCheckpointerDataInterface();
   void initializeGatherScatter();
   void initializeLayerIO(bool clobberFlag);

   bool isRoot() { return mFileManager->isRoot(); }

   void readInternal(double &timestamp, bool checkTimestampConsistency);

  private:
   std::shared_ptr<FileManager const> mFileManager = nullptr;
   std::string mPath;
   PVLayerLoc mLayerLoc;
   bool mDataExtended;
   bool mFileExtended;
   bool mReadOnly;
   bool mVerifyWrites;

   int mIndex = 0;
   std::vector<float *> mDataLocations;

   std::unique_ptr<LayerBatchGatherScatter> mGatherScatter;
   std::unique_ptr<LayerIO> mLayerIO;

   // The following values are written during checkpointing.
   // It would be more logical to write the value of mIndex, but for reasons of
   // backward compatibility, we continue to write the old values.
   int mNumFrames = 0; // number of batch elements handled by an MPIBlock;
   // that is, Index * MPIBlock->BatchDimension * loc->nbatch;
   long mFileStreamReadPos  = 0L; // Input file position of the LayerIO's FileStream
   long mFileStreamWritePos = 0L; // Output file position of the LayerIO's FileStream
};

} // namespace PV

#endif // LAYERFILE_HPP_
